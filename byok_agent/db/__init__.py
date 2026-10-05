"""Shared database adapters: Postgres (production) and SQLite (local dev)."""
from .base import Database
from .postgres import PostgresDB
from .schema import apply_schema
from .sqlite import SqliteDB

__all__ = ["Database", "PostgresDB", "SqliteDB", "apply_schema"]
