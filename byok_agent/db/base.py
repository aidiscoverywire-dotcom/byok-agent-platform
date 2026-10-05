"""Database access shared by the vault and the ledger.

All SQL in this project is written in Postgres placeholder style
(``$1``, ``$2``, ...). The SQLite adapter translates placeholders to ``?``,
so there is exactly one SQL dialect to maintain.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Database(Protocol):
    """Minimal async DB contract used by vault and ledger."""

    async def execute(self, sql: str, *args: Any) -> None: ...
    async def execute_rowcount(self, sql: str, *args: Any) -> int:
        """Execute a write and return the number of rows affected.

        Needed for atomic claim patterns (``UPDATE ... WHERE used=0``):
        the caller must distinguish "claimed" from "already taken", which
        ``execute`` (returning None) cannot express.
        """
        ...
    async def fetchrow(self, sql: str, *args: Any) -> Mapping[str, Any] | None: ...
    async def fetch(self, sql: str, *args: Any) -> list[Mapping[str, Any]]: ...
    async def executescript(self, sql: str) -> None:
        """Run a multi-statement SQL script (schema files). No parameters."""
        ...
    async def close(self) -> None: ...
