"""Per-call metering ledger.

Every model call writes one row: which key, which provider/model, token
usage, and an estimated USD cost via ``litellm.completion_cost()``.

The cost column is an ESTIMATE — it is explicitly NOT billing truth
(provider invoices are). It exists for budgets, dashboards, and alerts.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..agent.budgets import BudgetExceeded
from ..db.base import Database

log = logging.getLogger(__name__)


__all__ = ["BudgetExceeded"]  # re-exported; single canonical class lives in budgets.py


@dataclass(frozen=True)
class LedgerRow:
    id: int | None
    key_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    estimated_cost_usd: float  # ESTIMATE — never billing truth
    created_at: str
    status: str = "ok"  # ok | failed | skipped
    error: str | None = None  # sanitized reason for failed/skipped attempts
    task_id: str | None = None  # agent thread that caused this spend
    user_id: str | None = None  # tenant owner of the run


def _bare_model_name(model: str) -> str:
    """Strip the ``provider/`` prefix for litellm's price table.

    litellm's ``model_cost`` keys are bare names (``gpt-4o-mini``), NOT
    ``openai/gpt-4o-mini`` — looking up the prefixed form silently returns
    0.0 and disables reservation-based overshoot protection.
    """
    return model.split("/", 1)[1] if "/" in model else model


# Conservative fallback pricing for models missing from litellm's price
# table (GPT-4-class rates). Unknown != free: a silent 0.0 estimate
# disables reservation-based overshoot protection (fail-open) and lets
# unknown models spend outside every budget. All values derived from this
# are labeled ESTIMATE, never billing truth.
FALLBACK_INPUT_USD_PER_TOKEN = 30.0 / 1_000_000
FALLBACK_OUTPUT_USD_PER_TOKEN = 60.0 / 1_000_000

#: Providers whose inference is local (no API spend) — genuinely $0.
LOCAL_PROVIDERS = frozenset({"ollama"})


def _fallback_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Conservative cost guess for a model with no price data.

    Local providers (ollama) are genuinely free; everything else gets the
    GPT-4-class fallback so budget gates stay meaningful.
    """
    from ..llm.client import provider_of  # local: avoids any import cycle

    if provider_of(model) in LOCAL_PROVIDERS:
        return 0.0
    return (
        input_tokens * FALLBACK_INPUT_USD_PER_TOKEN
        + output_tokens * FALLBACK_OUTPUT_USD_PER_TOKEN
    )


def _usage_of(completion_response: Any) -> tuple[int, int]:
    """Best-effort (input, output) token counts from a provider response."""
    try:
        usage = getattr(completion_response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        return in_tok, out_tok
    except (TypeError, ValueError, AttributeError):
        return 0, 0


def estimate_max_cost_usd(
    model: str, est_input_tokens: int, max_output_tokens: int
) -> float:
    """Conservative upper-bound cost for budget reservation.

    Uses litellm's per-token price table when available. When the model has
    NO price data (unknown model), falls back to conservative GPT-4-class
    pricing instead of a silent 0.0 — a 0.0 estimate disables
    reservation-based overshoot protection (fail-open). Local providers
    (ollama) stay genuinely $0.
    """
    try:
        import litellm

        bare = _bare_model_name(model)
        info = litellm.model_cost.get(bare) or litellm.model_cost.get(model) or {}
        if not info:
            try:
                mi = litellm.get_model_info(bare)
                info = {
                    "input_cost_per_token": mi.get("input_cost_per_token", 0),
                    "output_cost_per_token": mi.get("output_cost_per_token", 0),
                }
            except Exception:
                info = {}
        per_in = float(info.get("input_cost_per_token", 0) or 0)
        per_out = float(info.get("output_cost_per_token", 0) or 0)
        if per_in > 0 or per_out > 0:
            return est_input_tokens * per_in + max_output_tokens * per_out
    except Exception as exc:
        log.warning("price table unavailable for %r: %s", model, exc)
    # No price data: conservative fallback (never a silent 0.0).
    log.warning(
        "no price data for model %r; using conservative fallback pricing "
        "(estimate, not billing truth)",
        model,
    )
    return _fallback_cost_usd(model, est_input_tokens, max_output_tokens)


def estimate_cost(model: str, completion_response: Any) -> float:
    """USD estimate via litellm.completion_cost(); conservative fallback if
    uncomputable (same reasoning as :func:`estimate_max_cost_usd`: unknown
    models must not ledger as free)."""
    try:
        import litellm

        # Try the response's own model name first, then the bare name —
        # litellm's price table keys are unprefixed.
        try:
            cost = float(
                litellm.completion_cost(completion_response=completion_response)
            )
        except Exception:
            cost = float(
                litellm.completion_cost(
                    completion_response=completion_response,
                    model=_bare_model_name(model),
                )
            )
        if cost > 0:
            return cost
    except Exception as exc:  # unknown model, no pricing data, litellm missing
        log.warning("completion_cost unavailable for %r: %s", model, exc)
    in_tok, out_tok = _usage_of(completion_response)
    if in_tok or out_tok:
        log.warning(
            "no price data for model %r; using conservative fallback pricing "
            "(estimate, not billing truth)",
            model,
        )
        return _fallback_cost_usd(model, in_tok, out_tok)
    return 0.0


class Ledger:
    """Append-only metering ledger over any :class:`Database`.

    Also tracks per-key cost RESERVATIONS (in-memory, process-wide): the
    :class:`~byok_agent.agent.budgets.BudgetEnforcer` reserves headroom
    before each model call and releases it after, so concurrent tasks
    sharing one key cannot each slip under the key's cap.

    ``key_budgets`` (optional :class:`~byok_agent.ops.budgets.KeyBudgetStore`):
    when attached, every recorded call accrues into the key's DAILY budget
    (estimates). Without this wiring the dashboard's daily caps were
    display-only — settable but never enforced or even accrued.
    """

    def __init__(self, db: Database, key_budgets: Any = None) -> None:
        self._db = db
        self._lock = asyncio.Lock()
        self._key_reserved: dict[str, float] = {}
        self._key_budgets = key_budgets

    async def record(
        self,
        *,
        key_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        estimated_cost_usd: float,
        status: str = "ok",
        error: str | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
    ) -> LedgerRow:
        """Write one metering row. Cost must come from :func:`estimate_cost`.

        ``status`` is ``"ok"`` for served calls, ``"failed"`` for attempts
        that raised, ``"skipped"`` for fallbacks never attempted — every
        attempt is metered, not just the successes.
        """
        created = datetime.now(timezone.utc).isoformat()
        row = await self._db.fetchrow(
            "INSERT INTO ledger (key_id, provider, model, input_tokens,"
            " output_tokens, cached_tokens, estimated_cost_usd, created_at,"
            " status, error, task_id, user_id)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)"
            " RETURNING id",
            key_id,
            provider,
            model,
            input_tokens,
            output_tokens,
            cached_tokens,
            estimated_cost_usd,
            created,
            status,
            error,
            task_id,
            user_id,
        )
        # SQLite has no RETURNING on older builds — fall back to last row.
        row_id = None
        if row is not None:
            row_id = int(row["id"])
        else:
            last = await self._db.fetchrow(
                "SELECT id FROM ledger WHERE key_id=$1 ORDER BY id DESC LIMIT 1",
                key_id,
            )
            row_id = int(last["id"]) if last else None
        # Accrue into the key's DAILY budget (best-effort: metering must
        # never break because the budget store had a hiccup). Zero-cost
        # rows (failed/skipped attempts) are skipped — no pointless writes.
        if self._key_budgets is not None and estimated_cost_usd > 0:
            try:
                await self._key_budgets.add_spend(key_id, estimated_cost_usd)
            except Exception as exc:  # never break metering
                log.warning("daily-budget accrue failed for %s: %s", key_id, exc)
        return LedgerRow(
            id=row_id,
            key_id=key_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            estimated_cost_usd=estimated_cost_usd,
            created_at=created,
            status=status,
            error=error,
            task_id=task_id,
            user_id=user_id,
        )

    async def get_spend(self, key_id: str) -> float:
        """Total ESTIMATED spend for one key (never billing truth)."""
        row = await self._db.fetchrow(
            "SELECT COALESCE(SUM(estimated_cost_usd),0) AS total"
            " FROM ledger WHERE key_id=$1",
            key_id,
        )
        return float(row["total"]) if row else 0.0

    async def check_key_budget(self, key_id: str, max_usd: float) -> None:
        """Raise :class:`BudgetExceeded` if the key's estimated spend hit the cap."""
        spent = await self.get_spend(key_id)
        if spent >= max_usd:
            raise BudgetExceeded(
                f"key {key_id!r} estimated spend ${spent:.4f} hit cap ${max_usd:.4f}"
                " (estimate, not billing truth)"
            )

    async def check_daily_cap(self, key_id: str, additional_usd: float = 0.0) -> None:
        """Enforce the dashboard-managed DAILY cap for a key.

        No cap row => unlimited (no-op). Raises :class:`BudgetExceeded`
        (translated from the store's ``KeyBudgetExceeded`` so existing
        ``except BudgetExceeded`` handlers keep working) when today's
        accrued spend plus ``additional_usd`` would exceed the cap. The
        :class:`~byok_agent.agent.budgets.BudgetEnforcer` calls this inside
        its gate so the dashboard's caps are actually enforced, not just
        displayed.
        """
        if self._key_budgets is None:
            return
        from ..ops.budgets import KeyBudgetExceeded

        try:
            await self._key_budgets.check(key_id, additional_usd)
        except KeyBudgetExceeded as exc:
            raise BudgetExceeded(str(exc)) from exc

    # -- per-key reservations (concurrent tasks sharing one key) ---------
    async def reserve_key(self, key_id: str, est_usd: float, cap_usd: float) -> None:
        """Atomically reserve ``est_usd`` of headroom against a key's cap.

        The check accounts for COMMITTED spend plus OUTSTANDING reservations
        from other in-flight tasks — not just committed spend — closing the
        race where concurrent tasks all pass the gate and overshoot N×.

        Raises :class:`BudgetExceeded` WITHOUT reserving on failure.
        Pair every successful call with :meth:`release_key_reservation`.
        """
        async with self._lock:
            spent = await self._get_spend_locked(key_id)
            reserved = self._key_reserved.get(key_id, 0.0)
            if spent + reserved + est_usd > cap_usd:
                raise BudgetExceeded(
                    f"key {key_id!r}: committed ${spent:.4f} + reserved "
                    f"${reserved:.4f} + est ${est_usd:.4f} > cap ${cap_usd:.4f} "
                    "(estimates)"
                )
            self._key_reserved[key_id] = reserved + est_usd

    async def release_key_reservation(self, key_id: str, est_usd: float) -> None:
        """Release headroom previously held by :meth:`reserve_key`."""
        async with self._lock:
            self._key_reserved[key_id] = max(
                0.0, self._key_reserved.get(key_id, 0.0) - est_usd
            )

    async def _get_spend_locked(self, key_id: str) -> float:
        row = await self._db.fetchrow(
            "SELECT COALESCE(SUM(estimated_cost_usd),0) AS total"
            " FROM ledger WHERE key_id=$1",
            key_id,
        )
        return float(row["total"]) if row else 0.0

    async def recent(self, key_id: str, limit: int = 20) -> list[LedgerRow]:
        """Newest-first metering rows for a key (dashboard/debug use)."""
        rows = await self._db.fetch(
            "SELECT * FROM ledger WHERE key_id=$1 ORDER BY id DESC LIMIT $2",
            key_id,
            limit,
        )
        return [
            LedgerRow(
                id=int(r["id"]),
                key_id=r["key_id"],
                provider=r["provider"],
                model=r["model"],
                input_tokens=int(r["input_tokens"]),
                output_tokens=int(r["output_tokens"]),
                cached_tokens=int(r["cached_tokens"]),
                estimated_cost_usd=float(r["estimated_cost_usd"]),
                created_at=r["created_at"],
                status=r.get("status", "ok") or "ok",
                error=r.get("error"),
            )
            for r in rows
        ]

    async def close(self) -> None:
        await self._db.close()
