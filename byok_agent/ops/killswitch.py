"""Persistent global kill switch.

The Phase 0 :func:`~byok_agent.agent.budgets.kill_switch_engaged` checks the
``AGENT_KILL_SWITCH`` env var / flag file. This module adds the DB-backed
flag the dashboard toggles: one row (``id=1``) in the ``kill_switch`` table.
:class:`~byok_agent.agent.budgets.BudgetEnforcer` consults BOTH — env first
(env can never be un-set by a dashboard bug), then the DB flag.

The DB read is cached for at most ``CACHE_TTL_S`` (5s) so the gate stays
cheap on hot paths.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

CACHE_TTL_S = 5.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class KillSwitch:
    """Read/write the persistent kill-switch row."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def is_engaged(self) -> bool:
        row = await self._db.fetchrow(
            "SELECT engaged FROM kill_switch WHERE id = 1"
        )
        return bool(row and row["engaged"])

    async def engage(self, *, by: str, reason: str = "") -> None:
        now = _utcnow()
        # $N placeholders may be reused in the ON CONFLICT clause: both
        # adapters bind $N by number, so each argument is passed ONCE and
        # the same SQL works on Postgres and SQLite.
        await self._db.execute(
            "INSERT INTO kill_switch (id, engaged, engaged_by, reason, engaged_at)"
            " VALUES (1, 1, $1, $2, $3)"
            " ON CONFLICT (id) DO UPDATE SET"
            " engaged = 1, engaged_by = $1, reason = $2, engaged_at = $3",
            by,
            reason,
            now,
        )

    async def disengage(self, *, by: str) -> None:
        now = _utcnow()
        # $N reused in the ON CONFLICT clause (see engage() note): each
        # argument passed once, works on both backends.
        await self._db.execute(
            "INSERT INTO kill_switch (id, engaged, engaged_by, reason, engaged_at)"
            " VALUES (1, 0, $1, '', $2)"
            " ON CONFLICT (id) DO UPDATE SET"
            " engaged = 0, engaged_by = $1, reason = '', engaged_at = $2",
            by,
            now,
        )

    async def state(self) -> dict:
        row = await self._db.fetchrow("SELECT * FROM kill_switch WHERE id = 1")
        if not row:
            return {"engaged": False, "engaged_by": None, "reason": "",
                    "engaged_at": None}
        return {
            "engaged": bool(row["engaged"]),
            "engaged_by": row["engaged_by"],
            "reason": row["reason"],
            "engaged_at": row["engaged_at"],
        }


def cached_checker(db: Any, ttl: float = CACHE_TTL_S) -> Callable[[], Awaitable[bool]]:
    """Build the async checker :class:`BudgetEnforcer` accepts.

    Result is cached for ``ttl`` seconds; a fresh read happens at most once
    per ``ttl`` window no matter how many gates fire.
    """
    ks = KillSwitch(db)
    # "at" starts at -inf so the FIRST check always reads the database.
    # (The old 0.0 assumed monotonic() >> ttl at startup; on a fresh boot
    # or container with uptime < ttl the gate served a stale False for the
    # first ttl seconds without ever consulting the DB.)
    cached: dict[str, Any] = {"at": float("-inf"), "value": False}

    async def check() -> bool:
        now = time.monotonic()
        if now - cached["at"] < ttl:
            return bool(cached["value"])
        value = await ks.is_engaged()
        cached["at"] = now
        cached["value"] = value
        return value

    # Exposed for tests: force a re-read on next gate.
    def _invalidate() -> None:
        cached["at"] = float("-inf")

    check.invalidate = _invalidate  # type: ignore[attr-defined]
    return check
