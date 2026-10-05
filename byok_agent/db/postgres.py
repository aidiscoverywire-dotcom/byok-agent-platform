"""Postgres database adapter (asyncpg connection pool)."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

import asyncpg

from .base import Database


class PostgresDB:
    """A :class:`Database` backed by an asyncpg pool."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _pool_or_raise(self) -> asyncpg.Pool:
        # Double-checked locking: pool creation is not free, and two
        # concurrent first-calls must not build two pools.
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(self._dsn)
        assert self._pool is not None
        return self._pool

    async def execute(self, sql: str, *args: Any) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(sql, *args)

    async def execute_rowcount(self, sql: str, *args: Any) -> int:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            # asyncpg returns a command tag like "UPDATE 1" or "DELETE 1"
            # ("INSERT 0 1" for inserts with an OID column).
            status = await conn.execute(sql, *args)
        try:
            return int(status.split()[-1])
        except (ValueError, IndexError, AttributeError):
            return 0

    async def fetchrow(self, sql: str, *args: Any) -> Mapping[str, Any] | None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
            return dict(row) if row is not None else None

    async def fetch(self, sql: str, *args: Any) -> list[Mapping[str, Any]]:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(r) for r in rows]

    async def executescript(self, sql: str) -> None:
        # asyncpg's execute() runs the whole script as one simple-protocol
        # query (no parameters allowed) — no naive ";" splitting, which
        # breaks on function bodies and string literals.
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(sql)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
