"""HMAC-chained audit log for everything the WhatsApp worker does.

Each row commits to the previous row's HMAC: ``hmac = HMAC(key,
seq|ts|actor|action|args_hash|prev_hash)``. Verifying the chain detects
any single-row edit or deletion in the middle.

HONEST LIMITATION (flagged in the user's audit, documented here on
purpose): if the HMAC key lives on the same host as the database, an
attacker with DB access can rebuild the ENTIRE chain from scratch with a
fresh key — the chain is tamper-evident against row edits, NOT against
full-chain forgery. The upgrade path is an external anchor: periodically
write the latest HMAC to an external store the DB attacker can't reach
(append-only log, a second host, or even a pinned message). Until that
exists, treat this log as "tamper-evident", not "tamper-proof".
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any

from ..db.base import Database

log = logging.getLogger(__name__)


def _is_conflict(exc: Exception) -> bool:
    """True for unique/primary-key violations (both backends).

    sqlite3 raises IntegrityError, asyncpg raises UniqueViolationError —
    matched by class name and message so neither backend needs importing.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name in ("IntegrityError", "UniqueViolationError")
        or "unique constraint" in msg
        or "duplicate key" in msg
        or "primary key" in msg
    )


class AuditLog:
    """Append-only, hash-chained audit trail."""

    def __init__(self, db: Database, hmac_key: bytes) -> None:
        if not hmac_key:
            raise ValueError("audit HMAC key is required")
        self._db = db
        self._key = hmac_key

    def _row_hmac(
        self, seq: int, ts: str, actor: str, action: str, args_hash: str, prev: str
    ) -> str:
        body = "|".join(
            [str(seq), ts, actor, action, args_hash, prev]
        ).encode()
        return hmac.new(self._key, body, hashlib.sha256).hexdigest()

    @staticmethod
    def hash_args(args: Any) -> str:
        """Stable hash of tool args (never stores raw secrets)."""
        try:
            canonical = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            canonical = str(args)
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def append(self, *, actor: str, action: str, args: Any = None) -> int:
        """Append one event. Returns the sequence number.

        ``seq`` is computed as max+1, so two writers appending in the same
        instant can collide on the primary key. On a unique-conflict the
        append is retried (up to 3 attempts) with a freshly recomputed
        ``seq`` — the chain stays linear because each retry re-reads the
        latest row for ``prev_hash``.
        """
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                return await self._append_once(
                    actor=actor, action=action, args=args
                )
            except Exception as exc:
                if not _is_conflict(exc):
                    raise
                last_exc = exc
                log.debug("audit append seq conflict; retrying")
        raise last_exc  # pragma: no cover — only if 3 collisions in a row

    async def _append_once(
        self, *, actor: str, action: str, args: Any = None
    ) -> int:
        """Single append attempt (no retry)."""
        last = await self._db.fetchrow(
            "SELECT seq, hmac FROM audit_log ORDER BY seq DESC LIMIT 1"
        )
        seq = int(last["seq"]) + 1 if last else 1
        prev = str(last["hmac"]) if last else "GENESIS"
        ts = datetime.now(timezone.utc).isoformat()
        args_hash = self.hash_args(args)
        row_hmac = self._row_hmac(seq, ts, actor, action, args_hash, prev)
        await self._db.execute(
            "INSERT INTO audit_log (seq, ts, actor, action, args_hash, prev_hash, hmac)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7)",
            seq,
            ts,
            actor,
            action,
            args_hash,
            prev,
            row_hmac,
        )
        return seq

    async def verify(self, limit: int = 10_000) -> tuple[bool, str]:
        """Verify the chain. Returns (ok, detail).

        Detects edited rows and mid-chain deletions. Does NOT detect a
        full-chain rebuild (see module docstring) or deletion of the
        trailing rows without an external anchor.
        """
        rows = await self._db.fetch(
            "SELECT seq, ts, actor, action, args_hash, prev_hash, hmac"
            " FROM audit_log ORDER BY seq ASC LIMIT $1",
            limit,
        )
        prev = "GENESIS"
        for row in rows:
            seq = int(row["seq"])
            expected = self._row_hmac(
                seq,
                str(row["ts"]),
                str(row["actor"]),
                str(row["action"]),
                str(row["args_hash"]),
                prev,
            )
            if not hmac.compare_digest(expected, str(row["hmac"])):
                return False, f"row seq={seq}: HMAC mismatch (edited?)"
            if str(row["prev_hash"]) != prev:
                return False, f"row seq={seq}: prev_hash break (deletion?)"
            prev = str(row["hmac"])
        return True, f"{len(rows)} rows verified"
