"""Schema application + idempotent migrations.

``apply_schema`` runs the backend's schema file as a whole (no naive
``";"`` splitting — that breaks on function bodies and string literals),
then applies idempotent ``ALTER TABLE`` migrations so existing dev
databases pick up new columns without a wipe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# (table, column, ddl) — applied idempotently after the base schema.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("vault_keys", "owner", "TEXT NOT NULL DEFAULT ''"),
    ("ledger", "status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("ledger", "error", "TEXT"),
    ("ledger", "task_id", "TEXT"),
    ("ledger", "user_id", "TEXT"),
    # Verify tokens move from the plaintext whatsapp_config column into the
    # vault; existing installs keep working via the legacy column.
    ("whatsapp_config", "verify_token_key_id", "TEXT"),
)


async def apply_schema(db: Any, backend: str) -> None:
    """Create tables (idempotent) and run pending column migrations."""
    schema_file = "schema.sql" if backend == "postgres" else "schema_sqlite.sql"
    sql = (Path(__file__).parent / schema_file).read_text(encoding="utf-8")
    await db.executescript(sql)
    for table, column, ddl in _MIGRATIONS:
        await _ensure_column(db, backend, table, column, ddl)


async def _ensure_column(
    db: Any, backend: str, table: str, column: str, ddl: str
) -> None:
    if backend == "postgres":
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl}"
        )
        return
    # SQLite has no ADD COLUMN IF NOT EXISTS — swallow the duplicate error.
    try:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except Exception as exc:
        if "duplicate column name" not in str(exc).lower():
            raise
        log.debug("column %s.%s already exists", table, column)
