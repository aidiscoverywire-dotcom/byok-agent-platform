"""Shared fixtures. No network calls in any test — litellm is mocked/faked."""
from __future__ import annotations

import os

# Strip proxy env vars: some sandboxes export malformed proxy config that
# breaks httpx at import time (litellm imports httpx). Tests never need a proxy.
for _v in (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(_v, None)

import pytest
from cryptography.fernet import Fernet


@pytest.fixture()
def master_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture()
def vault_env(monkeypatch, master_key: str):
    """Point APP_VAULT_KEY at a fresh random key for the test."""
    monkeypatch.setenv("APP_VAULT_KEY", master_key)
    monkeypatch.delenv("APP_VAULT_KEY_PREVIOUS", raising=False)
    monkeypatch.delenv("AGENT_KILL_SWITCH", raising=False)
    monkeypatch.delenv("AGENT_KILL_SWITCH_FILE", raising=False)
    return master_key


@pytest.fixture()
async def db(tmp_path):
    from byok_agent.db import apply_schema
    from byok_agent.db.sqlite import SqliteDB

    d = SqliteDB(tmp_path / "test.db")
    await apply_schema(d, "sqlite")
    yield d
    await d.close()


@pytest.fixture()
async def vault(db, vault_env):
    from byok_agent.vault import FernetVault

    v = FernetVault(db)
    yield v
    # db closed by the db fixture


@pytest.fixture()
async def ledger(db):
    from byok_agent.ledger import Ledger

    return Ledger(db)
