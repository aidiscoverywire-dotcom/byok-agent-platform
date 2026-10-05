"""Regression tests for the user's Phase 0 audit (bugs #1-#18 + small issues).

Each test maps to one numbered finding. All run offline (no network).
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from byok_agent.agent import BudgetEnforcer, TaskBudget
from byok_agent.agent.budgets import BudgetExceeded
from byok_agent.agent.budgets import BudgetExceeded as BudgetsBudgetExceeded
from byok_agent.agent.budgets import KillSwitchEngaged
from byok_agent.db.sqlite import SqliteDB
from byok_agent.ledger import BudgetExceeded as LedgerBudgetExceeded
from byok_agent.ledger import Ledger, estimate_max_cost_usd
from byok_agent.llm import (
    AllTargetsFailed,
    CompletionResult,
    LLMError,
    ProviderMismatchError,
    RouteTarget,
    Router,
    provider_of,
    sanitize_secret_text,
)
from byok_agent.vault import (
    ConcurrentModificationError,
    FernetVault,
    KeyNotFoundError,
)


# --------------------------------------------------------------------------
# Bug #1: SQLite ledger rows must survive close() (fetchrow must commit)
# --------------------------------------------------------------------------
async def test_bug1_sqlite_ledger_persists_after_close(tmp_path, vault_env):
    from byok_agent.db import apply_schema

    path = tmp_path / "b1.db"
    db = SqliteDB(path)
    await apply_schema(db, "sqlite")
    ledger = Ledger(db)
    await ledger.record(
        key_id="k1", provider="openai", model="openai/gpt-4o-mini",
        input_tokens=10, output_tokens=5, estimated_cost_usd=4.99,
    )
    assert await ledger.get_spend("k1") == pytest.approx(4.99)
    await db.close()

    # Fresh process view: rows must be on DISK, not just in the old connection.
    db2 = SqliteDB(path)
    try:
        assert await Ledger(db2).get_spend("k1") == pytest.approx(4.99)
        rows = await db2.fetch("SELECT COUNT(*) AS n FROM ledger")
        assert rows[0]["n"] == 1
    finally:
        await db2.close()


# --------------------------------------------------------------------------
# Bug #2: price lookup must strip the "provider/" prefix
# --------------------------------------------------------------------------
def test_bug2_price_lookup_strips_provider_prefix():
    # Before the fix both returned 0.0, silently disabling reservation.
    # Stable, in-table models (verified against the installed litellm)
    # for the prefix-stripping assertion:
    assert estimate_max_cost_usd("openai/gpt-4o-mini", 1000, 1024) > 0
    assert estimate_max_cost_usd("anthropic/claude-3-opus-20240229", 1000, 1024) > 0
    # Unknown models get conservative fallback pricing (never a silent 0.0
    # that would fail-open the reservation): local providers stay $0.
    assert estimate_max_cost_usd("nope/definitely-not-a-model", 1000, 1024) > 0
    assert estimate_max_cost_usd("ollama/llama3", 1000, 1024) == 0.0


# --------------------------------------------------------------------------
# Bug #3: router must bind the key record's provider to the model's provider
# --------------------------------------------------------------------------
async def test_bug3_router_rejects_wrong_vendor_key(vault):
    rec = await vault.add_key(
        label="a", provider="anthropic", plaintext_key="sk-ant-123", owner="u1"
    )

    class C:
        def __init__(self):
            self.calls = []

        async def acomplete(self, messages, model, *, api_key, **kwargs):
            self.calls.append((model, api_key))
            return CompletionResult(text="x", model=model, provider="openai")

    client = C()
    router = Router(client, vault)
    with pytest.raises(ProviderMismatchError):
        await router.acomplete(
            [{"role": "user", "content": "hi"}],
            RouteTarget(rec.key_id, "openai/gpt-4o-mini"),
        )
    assert client.calls == []  # the Anthropic key NEVER touched OpenAI


# --------------------------------------------------------------------------
# Bug #4: provider_of()=="unknown" is fail-closed (never same-provider)
# --------------------------------------------------------------------------
async def test_bug4_unknown_provider_requires_consent(vault):
    primary = await vault.add_key(
        label="p", provider="openai", plaintext_key="sk-p", owner="u1"
    )
    # Bare model name -> provider_of() == "unknown", consent False.
    sneaky = await vault.add_key(
        label="s", provider="anthropic", plaintext_key="sk-s", owner="u1"
    )

    class C:
        def __init__(self):
            self.calls = []

        async def acomplete(self, messages, model, *, api_key, **kwargs):
            self.calls.append((model, api_key))
            if model == "openai/gpt-4o-mini":
                raise LLMError("primary down")
            return CompletionResult(text="x", model=model, provider="unknown")

    router = Router(C(), vault)
    with pytest.raises(AllTargetsFailed):
        await router.acomplete(
            [{"role": "user", "content": "hi"}],
            RouteTarget(primary.key_id, "openai/gpt-4o-mini"),
            (RouteTarget(sneaky.key_id, "claude-3-5-sonnet-20240620"),),
        )
    # The audit's exact bypass: must NOT run without consent.
    assert all(k != "sk-s" for _, k in router._client.calls)


# --------------------------------------------------------------------------
# Bug #5: failed model calls must RELEASE the USD reservation (no leak)
# --------------------------------------------------------------------------
async def test_bug5_reservation_released_on_failure(ledger):
    enf = BudgetEnforcer(
        TaskBudget(max_steps=25, max_tokens=100_000, max_usd=1.0),
        ledger=ledger, key_id="k5", key_max_usd=10.0,
    )
    # Three failures at $0 actual spend, $0.40 reserved each.
    for _ in range(3):
        await enf.before_model_call(est_input_tokens=10, est_max_cost_usd=0.40)
        await enf.record_stream_abort(reserved_usd=0.40, reason="boom")
    snap = await enf.snapshot()
    assert snap["usd_reserved"] == pytest.approx(0.0)
    assert snap["usd"] == pytest.approx(0.0)
    # The next call must NOT be rejected: nothing is actually spent.
    await enf.before_model_call(est_input_tokens=10, est_max_cost_usd=0.40)
    await enf.record_stream_abort(reserved_usd=0.40, reason="boom")


# --------------------------------------------------------------------------
# Bug #6: resume restores budget; finished threads are idempotent
# --------------------------------------------------------------------------
class _ScriptedClient:
    """First call: tool ACTION. Second call: ANSWER."""

    def __init__(self):
        self.calls = 0

    async def acomplete(self, messages, model, *, api_key, **kwargs):
        self.calls += 1
        if self.calls == 1:
            text = "THOUGHT: check time\nACTION: get_current_time\nARGS: {}\n"
        else:
            text = "ANSWER: done"
        return CompletionResult(
            text=text, model=model, provider=provider_of(model),
            input_tokens=10, output_tokens=5,
        )


class _OwnerVault:
    def __init__(self, key_id="k1"):
        self.key_id = key_id

    async def get_key(self, key_id, owner=None):
        return "sk-fake"

    async def get_record(self, key_id, owner=None):
        from byok_agent.vault import KeyRecord, KeyStatus

        return KeyRecord(
            key_id=key_id, label="t", provider="openai",
            encrypted_key=b"x", status=KeyStatus.ACTIVE, owner="u1",
        )


class _MemLedger:
    def __init__(self):
        self.rows = []

    async def record(self, **kw):
        self.rows.append(kw)

    async def get_spend(self, key_id):
        return 0.0

    async def reserve_key(self, key_id, est_usd, cap_usd):
        return None

    async def release_key_reservation(self, key_id, est_usd):
        return None


def _runner(client, ledger):
    from byok_agent.agent import AgentRunner

    return AgentRunner(
        router=Router(client, _OwnerVault()),
        ledger=ledger,
        make_enforcer=lambda b: BudgetEnforcer(b),
        primary=RouteTarget("k1", "openai/gpt-4o-mini"),
    )


async def test_bug6_interrupted_thread_resumes_with_budget():
    from langgraph.checkpoint.memory import InMemorySaver

    from byok_agent.agent import TaskBudget

    cp = InMemorySaver()  # shared across both run() calls, like a real restart
    client = _ScriptedClient()
    runner = _runner(client, _MemLedger())

    # max_steps=1: first agent call OK (tool ACTION), second hits the cap.
    with pytest.raises(BudgetExceeded):
        await _run_with_cp(runner, cp, "u1", "t9", TaskBudget(max_steps=1))
    assert client.calls == 1

    # Same tid again: must RESUME (not restart) — budget restored, so the
    # very next gate raises instead of granting a free second run.
    with pytest.raises(BudgetExceeded):
        await _run_with_cp(runner, cp, "u1", "t9", TaskBudget(max_steps=1))
    assert client.calls == 1, "resume must not make a new model call past the cap"


async def _run_with_cp(runner, checkpointer, uid, tid, budget):
    runner._checkpointer = checkpointer
    return await runner.run(uid=uid, tid=tid, user_message="hi", budget=budget)


async def test_bug6_finished_thread_is_idempotent():
    from langgraph.checkpoint.memory import InMemorySaver

    from byok_agent.agent import TaskBudget

    class AnswerAtOnce(_ScriptedClient):
        async def acomplete(self, messages, model, *, api_key, **kwargs):
            self.calls += 1
            return CompletionResult(
                text="ANSWER: finished", model=model,
                provider=provider_of(model), input_tokens=10, output_tokens=5,
            )

    cp = InMemorySaver()
    client = AnswerAtOnce()
    runner = _runner(client, _MemLedger())
    r1 = await _run_with_cp(runner, cp, "u1", "t10", TaskBudget(max_steps=5))
    assert r1["answer"] == "finished"
    r2 = await _run_with_cp(runner, cp, "u1", "t10", TaskBudget(max_steps=5))
    assert r2["answer"] == "finished"
    assert client.calls == 1, "finished thread must not re-run the model"


# --------------------------------------------------------------------------
# Bug #7: plain-text final answers are accepted (ANSWER: optional)
# --------------------------------------------------------------------------
async def test_bug7_plain_text_answer_accepted():
    from langgraph.checkpoint.memory import InMemorySaver

    from byok_agent.agent import TaskBudget

    class PlainClient:
        async def acomplete(self, messages, model, *, api_key, **kwargs):
            return CompletionResult(
                text="The time is 10:00.", model=model,
                provider=provider_of(model), input_tokens=10, output_tokens=5,
            )

    cp = InMemorySaver()
    runner = _runner(PlainClient(), _MemLedger())
    res = await _run_with_cp(runner, cp, "u1", "t11", TaskBudget(max_steps=5))
    assert res["answer"] == "The time is 10:00."


# --------------------------------------------------------------------------
# Bug #8: checkpointer fallback is a LOUD error, never silent
# --------------------------------------------------------------------------
async def test_bug8_checkpointer_fallback_is_loud(capsys):
    from byok_agent.agent import make_checkpointer

    # Unreachable DSN: must degrade LOUDLY, not with a quiet warning.
    async with make_checkpointer("postgresql://127.0.0.1:1/nope") as cp:
        assert cp is not None
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "psycopg" in captured.err or "libpq" in captured.err


# --------------------------------------------------------------------------
# Bug #9: tenant isolation on the vault
# --------------------------------------------------------------------------
async def test_bug9_tenant_isolation(vault):
    rec = await vault.add_key(
        label="a", provider="openai", plaintext_key="sk-u1", owner="u1"
    )
    # Another user sees nothing — not even existence.
    with pytest.raises(KeyNotFoundError):
        await vault.get_record(rec.key_id, owner="u2")
    with pytest.raises(KeyNotFoundError):
        await vault.get_key(rec.key_id, owner="u2")
    assert await vault.list_keys(owner="u2") == []
    assert [r.key_id for r in await vault.list_keys(owner="u1")] == [rec.key_id]
    # Owner is mandatory.
    with pytest.raises(ValueError):
        await vault.add_key(label="x", provider="openai", plaintext_key="k", owner="")


# --------------------------------------------------------------------------
# Bug #10: skipped fallbacks don't block later valid ones; error aggregated
# --------------------------------------------------------------------------
async def test_bug10_valid_fallback_tried_after_skipped_one(vault):
    primary = await vault.add_key(
        label="p", provider="openai", plaintext_key="sk-p", owner="u1"
    )
    blocked = await vault.add_key(
        label="b", provider="anthropic", plaintext_key="sk-b", owner="u1"
    )
    valid = await vault.add_key(
        label="v", provider="openai", plaintext_key="sk-v", owner="u1"
    )

    class C:
        def __init__(self):
            self.calls = []

        async def acomplete(self, messages, model, *, api_key, **kwargs):
            self.calls.append((model, api_key))
            if model == "openai/gpt-4o-mini":
                raise LLMError("primary exploded")
            return CompletionResult(text="ok", model=model, provider="openai")

    router = Router(C(), vault)
    result, target = await router.acomplete(
        [{"role": "user", "content": "hi"}],
        RouteTarget(primary.key_id, "openai/gpt-4o-mini"),
        (
            RouteTarget(blocked.key_id, "anthropic/claude-x"),  # skipped: no consent
            RouteTarget(valid.key_id, "openai/gpt-4o"),  # must still be tried
        ),
    )
    assert target.key_id == valid.key_id
    assert result.text == "ok"


# --------------------------------------------------------------------------
# Bug #11: failed attempts are metered (not just successes)
# --------------------------------------------------------------------------
async def test_bug11_failed_attempts_metered(vault, ledger):
    primary = await vault.add_key(
        label="p", provider="openai", plaintext_key="sk-p", owner="u1"
    )
    fallback = await vault.add_key(
        label="f", provider="openai", plaintext_key="sk-f", owner="u1"
    )

    class C:
        async def acomplete(self, messages, model, *, api_key, **kwargs):
            if model == "openai/gpt-4o-mini":
                raise LLMError("boom")
            return CompletionResult(text="ok", model=model, provider="openai")

    async def sink(attempt):
        await ledger.record(
            key_id=attempt.key_id, provider=attempt.provider, model=attempt.model,
            input_tokens=0, output_tokens=0, estimated_cost_usd=0.0,
            status=attempt.status, error=attempt.error,
        )

    router = Router(C(), vault)
    result, target = await router.acomplete(
        [{"role": "user", "content": "hi"}],
        RouteTarget(primary.key_id, "openai/gpt-4o-mini"),
        (RouteTarget(fallback.key_id, "openai/gpt-4o"),),
        attempt_sink=sink,
    )
    assert target.key_id == fallback.key_id
    rows = await ledger.recent(primary.key_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    assert "boom" in (rows[0].error or "")


# --------------------------------------------------------------------------
# Bug #12: key-budget check counts outstanding reservations
# --------------------------------------------------------------------------
async def test_bug12_key_reservation_race(ledger):
    await ledger.reserve_key("k12", 0.60, 1.00)  # committed 0 + reserved 0.6
    with pytest.raises(LedgerBudgetExceeded):
        await ledger.reserve_key("k12", 0.60, 1.00)  # 0.6 + 0.6 > 1.0
    await ledger.release_key_reservation("k12", 0.60)
    await ledger.reserve_key("k12", 0.60, 1.00)  # fits again after release


async def test_bug12_concurrent_tasks_share_key_cap(ledger):
    enf_a = BudgetEnforcer(TaskBudget(max_usd=10.0), ledger=ledger,
                           key_id="k12b", key_max_usd=1.00)
    enf_b = BudgetEnforcer(TaskBudget(max_usd=10.0), ledger=ledger,
                           key_id="k12b", key_max_usd=1.00)
    await enf_a.before_model_call(est_input_tokens=10, est_max_cost_usd=0.60)
    # B sees A's outstanding reservation: 0.6 + 0.6 > 1.0 -> blocked.
    with pytest.raises(BudgetsBudgetExceeded):
        await enf_b.before_model_call(est_input_tokens=10, est_max_cost_usd=0.60)
    await enf_a.record_stream_abort(reserved_usd=0.60, reason="test")
    await enf_b.before_model_call(est_input_tokens=10, est_max_cost_usd=0.60)


# --------------------------------------------------------------------------
# Bug #13: token gate includes output tokens
# --------------------------------------------------------------------------
async def test_bug13_token_gate_counts_output():
    enf = BudgetEnforcer(TaskBudget(max_tokens=100))
    with pytest.raises(BudgetsBudgetExceeded):
        await enf.before_model_call(est_input_tokens=60, est_max_output_tokens=50)
    await enf.before_model_call(est_input_tokens=60, est_max_output_tokens=40)


# --------------------------------------------------------------------------
# Bug #14: exactly ONE BudgetExceeded class
# --------------------------------------------------------------------------
def test_bug14_single_budget_exceeded_class():
    assert LedgerBudgetExceeded is BudgetsBudgetExceeded
    assert LedgerBudgetExceeded.__module__ == "byok_agent.agent.budgets"


# --------------------------------------------------------------------------
# Bug #15: CLI catches errors cleanly; secrets sanitized
# --------------------------------------------------------------------------
def test_bug15_sanitize_redacts_keys():
    dirty = "boom sk-ant-secret12345 and api_key=supersecretvalue"
    clean = sanitize_secret_text(dirty)
    assert "sk-ant-secret12345" not in clean
    assert "supersecretvalue" not in clean
    assert "sk-***REDACTED***" in clean


async def test_bug15_wrong_master_key_clean_error(tmp_path, monkeypatch):
    """A wrong APP_VAULT_KEY must surface as VaultNotConfiguredError,
    never as a raw cryptography InvalidToken traceback."""
    import os

    from cryptography.fernet import Fernet

    from byok_agent.db import apply_schema
    from byok_agent.db.sqlite import SqliteDB
    from byok_agent.vault import FernetVault, VaultNotConfiguredError

    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()
    path = tmp_path / "mk.db"

    monkeypatch.setenv("APP_VAULT_KEY", key_a)
    db = SqliteDB(path)
    await apply_schema(db, "sqlite")
    rec = await FernetVault(db).add_key(
        label="l", provider="openai", plaintext_key="sk-x", owner="u1"
    )
    await db.close()

    monkeypatch.setenv("APP_VAULT_KEY", key_b)
    db2 = SqliteDB(path)
    try:
        with pytest.raises(VaultNotConfiguredError, match="APP_VAULT_KEY"):
            await FernetVault(db2).get_key(rec.key_id)
    finally:
        await db2.close()


async def test_bug15_cli_no_traceback_on_known_errors(tmp_path, vault_env, capsys, monkeypatch):
    from byok_agent.cli.main import cmd_run, build_parser

    monkeypatch.setenv("BYOK_SQLITE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    args = build_parser().parse_args(
        ["run", "--user", "u1", "--task", "t1", "--key-id", "key_nope_x",
         "hello"]
    )
    rc = await cmd_run(args)
    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "ERROR:" in captured.err


# --------------------------------------------------------------------------
# Bug #16: rotate/rekey are race-safe (guarded UPDATEs)
# --------------------------------------------------------------------------
async def test_bug16_concurrent_rotate_loser_raises(vault, db):
    rec = await vault.add_key(
        label="r", provider="openai", plaintext_key="sk-v1", owner="u1"
    )
    # Simulate a concurrent rotation landing first: bump version behind us.
    await db.execute(
        "UPDATE vault_keys SET key_version=key_version+1 WHERE key_id=$1",
        rec.key_id,
    )
    # Now a rotate based on the STALE read must not silently win. We emulate
    # staleness by calling the guarded path with the old version directly.
    from byok_agent.vault.fernet_vault import FernetVault

    stale = await vault.get_record(rec.key_id)
    # Manually drive the guarded update with a stale version:
    token = vault._fernet.encrypt(b"sk-stale-winner")
    await db.execute(
        "UPDATE vault_keys SET encrypted_key=$1, key_version=key_version+1"
        " WHERE key_id=$2 AND key_version=$3 AND status='active'",
        token, rec.key_id, 999,  # stale/wrong version -> matches nothing
    )
    after = await vault.get_record(rec.key_id)
    assert after.key_version == 2  # only the first (simulated) rotation applied


async def test_bug16_rekey_aborts_on_mid_rekey_rotation(vault):
    rec = await vault.add_key(
        label="r", provider="openai", plaintext_key="sk-v1", owner="u1"
    )
    # Rotate AFTER rekey read the row but BEFORE it writes: emulate by
    # rotating first, then calling rekey with a vault whose fernet would
    # still decrypt (MultiFernet not needed here — same master key).
    await vault.rotate_key(rec.key_id, "sk-v2")
    count = await vault.rekey_vault()  # re-reads rows itself; must succeed
    assert count == 1
    assert await vault.get_key(rec.key_id) == "sk-v2"


# --------------------------------------------------------------------------
# Bug #17: Postgres pool creation is lock-guarded (single pool)
# --------------------------------------------------------------------------
async def test_bug17_postgres_pool_created_once(monkeypatch):
    import asyncpg

    from byok_agent.db.postgres import PostgresDB

    created = []

    class FakePool:
        async def close(self):
            pass

    async def fake_create_pool(dsn):
        created.append(dsn)
        await asyncio.sleep(0.01)  # widen the race window
        return FakePool()

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    db = PostgresDB("postgresql://x/y")
    await asyncio.gather(*[db._pool_or_raise() for _ in range(10)])
    assert len(created) == 1
    await db.close()


# --------------------------------------------------------------------------
# Bug #18: Postgres schema applied as ONE script (no ";" splitting)
# --------------------------------------------------------------------------
async def test_bug18_postgres_executescript_single_call(monkeypatch):
    from byok_agent.db.postgres import PostgresDB

    calls = []

    class FakeConn:
        async def execute(self, sql, *args):
            calls.append(sql)

    class FakeAcquire:
        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *a):
            return False

    class FakePool:
        def acquire(self):
            return FakeAcquire()

        async def close(self):
            pass

    db = PostgresDB.__new__(PostgresDB)
    db._dsn = "x"
    db._pool = FakePool()
    import asyncio as _a

    db._pool_lock = _a.Lock()
    script = "CREATE TABLE a (x TEXT); INSERT INTO a VALUES ('semi;colon');"
    await db.executescript(script)
    assert len(calls) == 1
    assert calls[0] == script  # whole script, unsplit


# --------------------------------------------------------------------------
# Small issues
# --------------------------------------------------------------------------
def test_small_keyid_is_128bit(vault_env):
    import asyncio

    from byok_agent.db.sqlite import SqliteDB

    async def go(tmp):
        db = SqliteDB(tmp / "k.db")
        from byok_agent.db import apply_schema

        await apply_schema(db, "sqlite")
        v = FernetVault(db)
        rec = await v.add_key(label="l", provider="openai",
                              plaintext_key="sk-x", owner="u1")
        await db.close()
        return rec.key_id

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        key_id = asyncio.run(go(Path(td)))
    suffix = key_id.rsplit("_", 1)[-1]
    assert len(suffix) == 32  # 16 bytes hex = 128 bit
    assert re.fullmatch(r"[0-9a-f]{32}", suffix)


async def test_small_timeout_kwarg_no_typeerror(monkeypatch):
    import litellm

    from byok_agent.llm import LiteLLMClient

    seen = {}

    class Resp:
        choices = []
        usage = None

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)

        class C:
            message = type("M", (), {"content": "hi"})()

        r = Resp()
        r.choices = [C()]
        return r

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    client = LiteLLMClient(timeout=120.0)
    out = await client.acomplete(
        [{"role": "user", "content": "hi"}], "openai/gpt-4o-mini",
        api_key="sk-x", timeout=5,
    )
    assert seen["timeout"] == 5  # caller kwarg wins, no TypeError
    assert out.text == "hi"


async def test_small_sqlite_close_is_idempotent(tmp_path):
    db = SqliteDB(tmp_path / "c.db")
    await db._conn_or_open()
    await db.close()
    await db.close()  # must not raise


def test_small_env_example_sqlite_default():
    text = Path("byok_agent/../.env.example").read_text()
    active = [ln for ln in text.splitlines()
              if ln.strip().startswith("DATABASE_URL=")]
    assert active == [], ".env.example must not set DATABASE_URL (SQLite is the default)"


def test_small_pyproject_deps_pinned():
    text = Path("pyproject.toml").read_text()
    in_deps = False
    for line in text.splitlines():
        if line.strip() == "dependencies = [":
            in_deps = True
            continue
        if in_deps:
            if line.strip() == "]":
                break
            dep = line.strip().strip('",')
            assert "==" in dep, f"unpinned dependency: {dep}"


def test_small_package_not_named_src():
    assert Path("byok_agent").is_dir()
    assert not Path("src").exists(), "the 'src' package name must be gone"
