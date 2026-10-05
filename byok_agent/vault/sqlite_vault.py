"""SQLite-backed vault for local dev — same interface as the Postgres vault."""
from __future__ import annotations

from pathlib import Path

from ..db.sqlite import SqliteDB
from .fernet_vault import FernetVault


class SqliteVault(FernetVault):
    """A :class:`FernetVault` over a local SQLite file.

    Args:
        path: SQLite file path (created if missing).
        apply_schema: when True, create ``vault_keys``/``ledger`` tables.
    """

    def __init__(self, path: str | Path, *, apply_schema: bool = True) -> None:
        self._path = Path(path)
        super().__init__(SqliteDB(self._path))
        self._apply_schema = apply_schema

    async def init(self) -> None:
        """Create tables from the SQLite schema (idempotent)."""
        if not self._apply_schema:
            return
        from ..db import apply_schema

        await apply_schema(self._db, "sqlite")
