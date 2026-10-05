"""Per-key daily USD budgets (estimates).

``key_budgets`` holds one row per key: a daily cap and the spend accrued
*today* (UTC). The day rolls over automatically — the first check on a new
UTC day resets ``spent_today`` and the alert flags.

Alerts: callers can ask for the alert state (80% / 100% reached) and the
dashboard surfaces it. Flags are latched per day so the operator is not
spammed; they reset with the day.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class KeyBudgetStore:
    def __init__(self, db: Any) -> None:
        self._db = db

    async def set_cap(self, key_id: str, daily_usd_cap: float) -> None:
        cap = float(daily_usd_cap)
        # $2 is reused in the ON CONFLICT clause; both adapters bind $N by
        # number, so the argument is passed once and the SQL is portable.
        await self._db.execute(
            "INSERT INTO key_budgets (key_id, daily_usd_cap, day) VALUES ($1, $2, $3)"
            " ON CONFLICT (key_id) DO UPDATE SET daily_usd_cap = $2",
            key_id,
            cap,
            _today(),
        )

    async def remove_cap(self, key_id: str) -> None:
        await self._db.execute("DELETE FROM key_budgets WHERE key_id = $1", key_id)

    async def list_caps(self) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT key_id, daily_usd_cap, day, spent_today, alerted_80,"
            " alerted_100 FROM key_budgets ORDER BY key_id"
        )
        return [dict(r) for r in rows]

    async def _roll_day(self, key_id: str) -> None:
        today = _today()
        # $1 is reused in the WHERE clause; both adapters bind $N by number,
        # so each argument is passed once.
        await self._db.execute(
            "UPDATE key_budgets SET day = $1, spent_today = 0,"
            " alerted_80 = 0, alerted_100 = 0"
            " WHERE key_id = $2 AND day <> $1",
            today,
            key_id,
        )

    async def add_spend(self, key_id: str, usd: float) -> dict:
        """Accrue spend; returns {'cap', 'spent_today', 'alert'}.

        ``alert`` is None | 'warn80' | 'cap100' — latched once per day.
        No cap row => {'cap': None, ...} (unlimited).
        """
        await self._roll_day(key_id)
        row = await self._db.fetchrow(
            "SELECT daily_usd_cap, spent_today, alerted_80, alerted_100"
            " FROM key_budgets WHERE key_id = $1",
            key_id,
        )
        if row is None:
            return {"cap": None, "spent_today": 0.0, "alert": None}
        spent = float(row["spent_today"]) + float(usd)
        cap = float(row["daily_usd_cap"])
        alert = None
        updates: list[str] = []
        if spent >= cap and not row["alerted_100"]:
            alert = "cap100"
            updates.append("alerted_100 = 1")
        elif spent >= 0.8 * cap and not row["alerted_80"]:
            alert = "warn80"
            updates.append("alerted_80 = 1")
        set_clause = ", ".join(["spent_today = $1"] + updates)
        await self._db.execute(
            f"UPDATE key_budgets SET {set_clause} WHERE key_id = $2",
            spent,
            key_id,
        )
        return {"cap": cap, "spent_today": spent, "alert": alert}

    async def check(self, key_id: str, additional_usd: float = 0.0) -> None:
        """Raise :class:`KeyBudgetExceeded` if the cap would be exceeded."""
        await self._roll_day(key_id)
        row = await self._db.fetchrow(
            "SELECT daily_usd_cap, spent_today FROM key_budgets WHERE key_id = $1",
            key_id,
        )
        if row is None:
            return
        if float(row["spent_today"]) + float(additional_usd) > float(
            row["daily_usd_cap"]
        ):
            raise KeyBudgetExceeded(
                f"key {key_id}: daily budget ${row['daily_usd_cap']:.2f}"
                f" would be exceeded (spent ${row['spent_today']:.2f} today,"
                " estimates)"
            )


class KeyBudgetExceeded(Exception):
    """A per-key daily USD budget cap was hit."""
