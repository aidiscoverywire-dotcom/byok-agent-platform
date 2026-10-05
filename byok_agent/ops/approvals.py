"""Persistent human-approval queue.

``interrupt()`` asks (build promotions, deploy gates) are recorded here so the
operator dashboard or CLI can approve/deny them out of band. Payloads are JSON
and must never contain secrets — only references (paths, diffs, key ids).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ApprovalRecord:
    id: int
    thread_id: str
    kind: str
    summary: str
    payload: dict
    status: str
    created_at: str
    decided_at: str | None
    decided_by: str | None


def _row_to_record(row: Any) -> ApprovalRecord:
    try:
        payload = json.loads(row["payload"] or "{}")
    except (ValueError, TypeError):
        payload = {}
    return ApprovalRecord(
        id=int(row["id"]),
        thread_id=row["thread_id"],
        kind=row["kind"],
        summary=row["summary"],
        payload=payload,
        status=row["status"],
        created_at=row["created_at"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
    )


class ApprovalStore:
    """DB-backed approval queue."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def request(
        self, *, thread_id: str, kind: str, summary: str, payload: dict
    ) -> int:
        row = await self._db.fetchrow(
            "INSERT INTO approval_queue (thread_id, kind, summary, payload,"
            " status, created_at) VALUES ($1, $2, $3, $4, 'pending', $5)"
            " RETURNING id",
            thread_id,
            kind,
            summary,
            json.dumps(payload),
            _utcnow(),
        )
        # SQLite adapter: fetchrow commits (INSERT...RETURNING), so the id
        # survives close(). Postgres RETURNING works natively.
        assert row is not None
        return int(row["id"])

    async def list_pending(self, limit: int = 50) -> list[ApprovalRecord]:
        rows = await self._db.fetch(
            "SELECT * FROM approval_queue WHERE status = 'pending'"
            " ORDER BY id ASC LIMIT $1",
            limit,
        )
        return [_row_to_record(r) for r in rows]

    async def get(self, approval_id: int) -> ApprovalRecord | None:
        row = await self._db.fetchrow(
            "SELECT * FROM approval_queue WHERE id = $1", approval_id
        )
        return _row_to_record(row) if row else None

    async def decide(
        self, approval_id: int, *, approved: bool, decided_by: str
    ) -> ApprovalRecord:
        status = "approved" if approved else "denied"
        row = await self._db.fetchrow(
            "UPDATE approval_queue SET status = $1, decided_at = $2,"
            " decided_by = $3 WHERE id = $4 AND status = 'pending'"
            " RETURNING *",
            status,
            _utcnow(),
            decided_by,
            approval_id,
        )
        if row is None:
            raise KeyError(
                f"approval {approval_id} not found or already decided"
            )
        return _row_to_record(row)
