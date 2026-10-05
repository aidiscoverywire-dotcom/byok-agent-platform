"""Per-task budget enforcement + global kill switch.

Race-safety design (fixes the parallel-overshoot bug): every gate and every
recording goes through ONE :class:`asyncio.Lock` held by the enforcer, and
``before_model_call`` *reserves* the step/token headroom atomically before
any branch proceeds. Parallel ``Send`` fan-out branches therefore share a
single counter — N branches cannot each pass the check and overshoot N×.

Gates run BEFORE every model call AND every tool execution.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """A per-task cap (steps / tokens / USD) was hit."""


class KillSwitchEngaged(Exception):
    """The global kill switch is on — no new work may start."""


@dataclass(frozen=True)
class TaskBudget:
    max_steps: int = 25
    max_tokens: int = 100_000
    max_usd: float = 5.0


def kill_switch_engaged() -> bool:
    """True if ``AGENT_KILL_SWITCH=1`` or the flag file exists."""
    if os.environ.get("AGENT_KILL_SWITCH", "").strip() == "1":
        return True
    flag = os.environ.get("AGENT_KILL_SWITCH_FILE", "").strip()
    return bool(flag) and Path(flag).exists()


class BudgetEnforcer:
    """Atomic per-task budget enforcement.

    Args:
        budget: Caps for this task.
        ledger: Optional :class:`~byok_agent.ledger.ledger.Ledger` for per-key spend checks.
        key_id: Key whose cumulative estimated spend is capped.
        key_max_usd: Cap for the key's total estimated spend (requires ledger).
        kill_switch_checker: Optional async callable returning True when the
            persistent (DB) kill switch is engaged. Checked on every gate in
            addition to the env-var kill switch. The checker from
            :func:`byok_agent.ops.killswitch.cached_checker` caches the DB
            read for <=5s.
    """

    def __init__(
        self,
        budget: TaskBudget,
        *,
        ledger: object | None = None,
        key_id: str | None = None,
        key_max_usd: float | None = None,
        kill_switch_checker: Any | None = None,
    ) -> None:
        self._budget = budget
        self._ledger = ledger
        self._key_id = key_id
        self._key_max_usd = key_max_usd
        self._kill_switch_checker = kill_switch_checker
        self._lock = asyncio.Lock()
        self._steps = 0
        self._tokens = 0
        self._usd = 0.0
        self._usd_reserved = 0.0  # cost headroom held by in-flight calls
        self._aborted_partial: list[dict] = []  # stream-abort partials

    # -- introspection (snapshot under lock) ------------------------------
    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "steps": self._steps,
                "tokens": self._tokens,
                "usd": round(self._usd, 6),
                "usd_reserved": round(self._usd_reserved, 6),
                "aborted_partials": len(self._aborted_partial),
            }

    async def restore(self, *, steps: int, tokens: int, usd: float) -> None:
        """Restore counters from a persisted checkpoint (crash-resume).

        Call exactly once on a FRESH enforcer before resuming a thread.
        In-flight reservations are zeroed: no call survived the crash, so
        no reservation did either.
        """
        async with self._lock:
            self._steps = int(steps)
            self._tokens = int(tokens)
            self._usd = float(usd)
            self._usd_reserved = 0.0

    # -- gates ------------------------------------------------------------
    def _check_kill(self) -> None:
        if kill_switch_engaged():
            raise KillSwitchEngaged(
                "global kill switch is engaged; refusing new work"
            )

    async def _check_kill_db(self) -> None:
        """Also consult the persistent (DB) kill switch, if wired.

        The env-var kill switch can never be un-set by a dashboard bug;
        the DB flag is the operator toggle. Both must be clear.
        """
        checker = self._kill_switch_checker
        if checker is not None and await checker():
            raise KillSwitchEngaged(
                "persistent kill switch is engaged (dashboard); refusing new work"
            )

    async def before_model_call(
        self,
        *,
        est_input_tokens: int = 0,
        est_max_output_tokens: int = 0,
        est_max_cost_usd: float = 0.0,
    ) -> None:
        """Gate BEFORE a model call; atomically reserves one step AND cost headroom.

        ``est_max_cost_usd`` should be a conservative upper bound for the
        call (see :func:`byok_agent.ledger.ledger.estimate_max_cost_usd`).
        Reserving it under the same lock is what bounds parallel-branch
        overshoot: branches that would push committed + reserved spend over
        the cap are rejected before they start.

        The per-KEY budget check also runs INSIDE this lock, against
        committed spend + outstanding key reservations (not just committed
        spend), so concurrent tasks sharing one key cannot each slip under
        the cap. If the key check fails, the task-level reservation made
        above is rolled back — no leak.

        Raises :class:`KillSwitchEngaged` / :class:`BudgetExceeded`.
        """
        async with self._lock:
            self._check_kill()
            await self._check_kill_db()
            if self._steps + 1 > self._budget.max_steps:
                raise BudgetExceeded(
                    f"step cap hit: {self._steps}/{self._budget.max_steps}"
                )
            if (
                self._tokens + est_input_tokens + est_max_output_tokens
                > self._budget.max_tokens
            ):
                raise BudgetExceeded(
                    f"token cap would be exceeded: "
                    f"{self._tokens}+{est_input_tokens}+{est_max_output_tokens}"
                    f"/{self._budget.max_tokens}"
                )
            if self._usd + self._usd_reserved + est_max_cost_usd > self._budget.max_usd:
                raise BudgetExceeded(
                    f"USD cap would be exceeded: committed ${self._usd:.4f} + "
                    f"reserved ${self._usd_reserved:.4f} + est "
                    f"${est_max_cost_usd:.4f} > ${self._budget.max_usd:.4f} "
                    "(estimates)"
                )
            # Reserve atomically so parallel branches can't all pass.
            self._steps += 1
            self._usd_reserved += est_max_cost_usd
            # Dashboard-managed DAILY cap for this key (no cap row =>
            # unlimited). Checked here — inside the same lock — so the
            # dashboard's per-key budgets are enforced, not just displayed.
            if self._ledger is not None and self._key_id is not None:
                try:
                    await self._ledger.check_daily_cap(  # type: ignore[attr-defined]
                        self._key_id, est_max_cost_usd
                    )
                except Exception:
                    self._steps -= 1
                    self._usd_reserved = max(
                        0.0, self._usd_reserved - est_max_cost_usd
                    )
                    raise
            if (
                self._ledger is not None
                and self._key_id is not None
                and self._key_max_usd is not None
            ):
                try:
                    await self._ledger.reserve_key(  # type: ignore[attr-defined]
                        self._key_id, est_max_cost_usd, self._key_max_usd
                    )
                except Exception:
                    # Roll back the task-level reservation: a failed gate
                    # must not leave headroom held.
                    self._steps -= 1
                    self._usd_reserved = max(
                        0.0, self._usd_reserved - est_max_cost_usd
                    )
                    raise

    async def before_tool_call(self, tool_name: str) -> None:
        """Gate BEFORE a tool execution (tools are free but not unlimited)."""
        async with self._lock:
            self._check_kill()
            await self._check_kill_db()
            if self._steps > self._budget.max_steps:
                raise BudgetExceeded(f"step cap hit: {self._steps}")
        log.debug("tool gate passed: %s", tool_name)

    # -- recording ----------------------------------------------------------
    async def record_model_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        reserved_usd: float = 0.0,
    ) -> None:
        """Record actual usage after a model call (atomic).

        Releases ``reserved_usd`` previously held by :meth:`before_model_call`
        — both the task-level reservation and the per-key reservation — and
        commits the actual cost.
        """
        async with self._lock:
            self._usd_reserved = max(0.0, self._usd_reserved - reserved_usd)
            self._tokens += input_tokens + output_tokens
            self._usd += cost_usd
            if self._usd > self._budget.max_usd:
                log.warning(
                    "task USD estimate $%.4f exceeded cap $%.4f — "
                    "no further calls will be gated through",
                    self._usd,
                    self._budget.max_usd,
                )
        if (
            self._ledger is not None
            and self._key_id is not None
            and self._key_max_usd is not None
        ):
            await self._ledger.release_key_reservation(  # type: ignore[attr-defined]
                self._key_id, reserved_usd
            )

    async def record_stream_abort(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        reserved_usd: float = 0.0,
        reason: str = "stream aborted",
    ) -> None:
        """Record PARTIAL usage when a streamed call is aborted.

        Aborted streams report no final usage object, so without this the
        cost would be silently under-recorded. The ``reserved_usd`` held by
        :meth:`before_model_call` is RELEASED here — a failed call must not
        leave headroom pinned forever (that was a real reservation leak).
        """
        async with self._lock:
            self._usd_reserved = max(0.0, self._usd_reserved - reserved_usd)
            self._tokens += input_tokens + output_tokens
            self._usd += cost_usd
            self._aborted_partial.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "reason": reason,
                }
            )
        if (
            self._ledger is not None
            and self._key_id is not None
            and self._key_max_usd is not None
        ):
            await self._ledger.release_key_reservation(  # type: ignore[attr-defined]
                self._key_id, reserved_usd
            )
        log.warning("recorded aborted-stream partial usage: %s", reason)

    async def over_usd_cap(self) -> bool:
        async with self._lock:
            return self._usd > self._budget.max_usd
