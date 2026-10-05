"""Sender pairing: who is allowed to drive the agent over WhatsApp.

Threat model: the business number is reachable by ANYONE on WhatsApp.
Without pairing, a stranger could burn the owner's LLM budget or run
tools. So:

- ``pending`` — first contact; the worker replies ONCE with pairing
  instructions, then stays silent for that sender (no agent run).
- ``paired`` — the sender proved ownership of a single-use code the
  owner generated via CLI; full access.
- ``blocked`` — silence. No reply, no agent run, nothing.

Codes are single-use, owner-scoped, and expire after 15 minutes. All
pairing checks happen BEFORE any agent/tool execution in the worker.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

from ..db.base import Database

log = logging.getLogger(__name__)

PAIRING_CODE_TTL_MINUTES = 15

NOT_PAIRED_TEXT = (
    "This number isn't paired with the agent yet.\n"
    "Ask the owner to run: byok whatsapp pair-code --user <owner>\n"
    "then send the code here within 15 minutes."
)


class PairingStore:
    """DB-backed pairing state over any :class:`Database`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- senders ---------------------------------------------------------
    async def sender_status(self, sender_id: str) -> str:
        """'paired' | 'pending' | 'blocked' | 'unknown'."""
        row = await self._db.fetchrow(
            "SELECT status FROM whatsapp_senders WHERE sender_id=$1", sender_id
        )
        return str(row["status"]) if row else "unknown"

    async def _upsert_sender(
        self, sender_id: str, owner: str, status: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = await self._db.fetchrow(
            "SELECT sender_id FROM whatsapp_senders WHERE sender_id=$1", sender_id
        )
        if existing:
            await self._db.execute(
                "UPDATE whatsapp_senders SET status=$1, owner=$2 WHERE sender_id=$3",
                status,
                owner,
                sender_id,
            )
        else:
            await self._db.execute(
                "INSERT INTO whatsapp_senders (sender_id, owner, status, paired_at)"
                " VALUES ($1,$2,$3,$4)",
                sender_id,
                owner,
                status,
                now if status == "paired" else None,
            )

    async def mark_pending(self, sender_id: str) -> None:
        await self._upsert_sender(sender_id, "", "pending")

    async def block_sender(self, sender_id: str, owner: str = "") -> None:
        await self._upsert_sender(sender_id, owner, "blocked")

    # -- codes -----------------------------------------------------------
    async def issue_code(self, owner: str) -> str:
        """Generate a single-use pairing code for ``owner``."""
        code = f"{secrets.randbelow(900000) + 100000:06d}"  # 6 digits
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO pairing_codes (code, owner, created_at, used)"
            " VALUES ($1,$2,$3,0)",
            code,
            owner,
            now,
        )
        return code

    async def redeem_code(self, sender_id: str, code: str) -> str | None:
        """Redeem a code. Returns the owner on success, None on failure.

        Single-use and 15-minute TTL. Expired/used/unknown codes fail
        closed. On success the sender becomes 'paired' under that owner.

        The claim is atomic: ``UPDATE ... WHERE used=0`` matches exactly
        one row, and the affected-row count decides the winner — two
        concurrent redeemers cannot both claim the same code.
        """
        code = code.strip()
        row = await self._db.fetchrow(
            "SELECT code, owner, created_at, used FROM pairing_codes WHERE code=$1",
            code,
        )
        if not row or int(row["used"]):
            return None
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
        except ValueError:
            return None
        if created.tzinfo is None:
            # Tolerate naive DB timestamps (e.g. TIMESTAMP WITHOUT TIME ZONE
            # columns): interpret as UTC instead of crashing on aware/naive
            # subtraction.
            created = created.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - created > timedelta(
            minutes=PAIRING_CODE_TTL_MINUTES
        ):
            return None
        owner = str(row["owner"])
        claimed = await self._db.execute_rowcount(
            "UPDATE pairing_codes SET used=1 WHERE code=$1 AND used=0", code
        )
        if claimed != 1:
            # Lost the race (or the row vanished): fail closed.
            log.warning("pairing code %s claim lost race; denying", code)
            return None
        await self._upsert_sender(sender_id, owner, "paired")
        log.info("sender %s paired to owner %s", sender_id, owner)
        return owner
