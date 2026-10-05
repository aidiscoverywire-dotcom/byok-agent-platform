"""SQLite database adapter (aiosqlite) for local dev without Postgres.

Translates Postgres ``$N`` placeholders to ``?``, mapping BY NUMBER
(``$N`` always binds the Nth argument — exactly like asyncpg), so the same
SQL and the same argument list run unchanged on both backends. Row access
is by column name via ``sqlite3.Row``.

Concurrency note: aiosqlite executes all statements on a single background
thread, so one shared connection is safe from a single event loop. Only
connection *creation* is guarded by a lock (double-checked).
"""
from __future__ import annotations

import asyncio
import atexit
import re
from pathlib import Path
from typing import Any, Mapping

import aiosqlite

from .base import Database

_PLACEHOLDER = re.compile(r"\$(\d+)")


def _translate(sql: str, args: tuple) -> tuple[str, tuple]:
    """Translate Postgres ``$N`` placeholders to ``?``, mapping by NUMBER.

    Each ``$N`` becomes ``?`` and the argument list is rebuilt so
    ``$N`` always binds ``args[N-1]`` — exactly like Postgres/asyncpg.
    Reused placeholders (``$1 ... $1``) therefore work with each argument
    passed ONCE; callers must NOT duplicate args for SQLite.

    Raises :class:`ValueError` if a placeholder has no matching argument
    (fail fast instead of a cryptic driver error).
    """
    new_args: list = []

    def repl(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(args):
            raise ValueError(
                f"placeholder ${m.group(1)} has no matching argument "
                f"({len(args)} supplied)"
            )
        new_args.append(args[idx])
        return "?"

    return _PLACEHOLDER.sub(repl, sql), tuple(new_args)


class SqliteDB:
    """A :class:`Database` backed by a local SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None
        self._open_lock = asyncio.Lock()
        # Last-resort close: unregistered again by a clean close().
        atexit.register(self._atexit_close)

    async def _conn_or_open(self) -> aiosqlite.Connection:
        if self._conn is None:
            async with self._open_lock:
                if self._conn is None:  # double-checked locking
                    conn = await aiosqlite.connect(self._path)
                    conn.row_factory = aiosqlite.Row
                    await conn.execute("PRAGMA journal_mode=WAL;")
                    await conn.commit()
                    self._conn = conn
        assert self._conn is not None
        return self._conn

    async def execute(self, sql: str, *args: Any) -> None:
        conn = await self._conn_or_open()
        qsql, qargs = _translate(sql, args)
        await conn.execute(qsql, qargs)
        await conn.commit()

    async def execute_rowcount(self, sql: str, *args: Any) -> int:
        conn = await self._conn_or_open()
        qsql, qargs = _translate(sql, args)
        cur = await conn.execute(qsql, qargs)
        n = cur.rowcount
        await cur.close()
        await conn.commit()
        return int(n) if n is not None and n >= 0 else 0

    async def fetchrow(self, sql: str, *args: Any) -> Mapping[str, Any] | None:
        # NOTE: INSERT...RETURNING flows through here, so we MUST commit —
        # otherwise ledger/vault rows vanish when the connection closes.
        conn = await self._conn_or_open()
        qsql, qargs = _translate(sql, args)
        cur = await conn.execute(qsql, qargs)
        row = await cur.fetchone()
        await cur.close()
        await conn.commit()
        return dict(row) if row is not None else None

    async def fetch(self, sql: str, *args: Any) -> list[Mapping[str, Any]]:
        conn = await self._conn_or_open()
        qsql, qargs = _translate(sql, args)
        cur = await conn.execute(qsql, qargs)
        rows = await cur.fetchall()
        await cur.close()
        await conn.commit()
        return [dict(r) for r in rows]

    async def executescript(self, sql: str) -> None:
        conn = await self._conn_or_open()
        await conn.executescript(sql)
        await conn.commit()

    async def close(self) -> None:
        async with self._open_lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
        # Unregister the atexit hook: a clean close needs no last resort.
        try:
            atexit.unregister(self._atexit_close)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _atexit_close(self) -> None:
        """Best-effort close at interpreter exit (sync context).

        aiosqlite's background thread can otherwise keep the process alive
        when user code forgets ``await db.close()``. If an event loop is
        still running we schedule the close; otherwise we drop the
        reference and let the daemon thread die with the process.
        """
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running():
            loop.create_task(conn.close())
