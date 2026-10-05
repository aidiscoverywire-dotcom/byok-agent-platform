"""Budget enforcer tests: pre-call gates, kill switch, parallel overshoot."""
from __future__ import annotations

import asyncio

import pytest

from byok_agent.agent.budgets import (
    BudgetEnforcer,
    BudgetExceeded,
    KillSwitchEngaged,
    TaskBudget,
    kill_switch_engaged,
)


def test_blocks_over_step_cap_before_model_call():
    async def go():
        enf = BudgetEnforcer(TaskBudget(max_steps=1, max_tokens=10**9, max_usd=10**9))
        await enf.before_model_call()  # step 1: ok
        with pytest.raises(BudgetExceeded):
            await enf.before_model_call()  # step 2: blocked BEFORE any call

    asyncio.run(go())


def test_blocks_over_token_cap_before_model_call():
    async def go():
        enf = BudgetEnforcer(TaskBudget(max_steps=100, max_tokens=50, max_usd=10**9))
        with pytest.raises(BudgetExceeded):
            await enf.before_model_call(est_input_tokens=51)

    asyncio.run(go())


def test_kill_switch_env(monkeypatch):
    async def go():
        enf = BudgetEnforcer(TaskBudget())
        with pytest.raises(KillSwitchEngaged):
            await enf.before_model_call()
        with pytest.raises(KillSwitchEngaged):
            await enf.before_tool_call("get_current_time")

    monkeypatch.setenv("AGENT_KILL_SWITCH", "1")
    assert kill_switch_engaged()
    asyncio.run(go())


def test_kill_switch_file(monkeypatch, tmp_path):
    flag = tmp_path / "kill"
    flag.write_text("stop")
    monkeypatch.setenv("AGENT_KILL_SWITCH_FILE", str(flag))
    assert kill_switch_engaged()


def test_parallel_branches_cannot_overshoot_usd_cap():
    """8 concurrent consumers vs a $1.00 cap: total committed cost must stay ≤ cap.

    Each consumer reserves $0.20 before its fake call and commits $0.20
    after. Without atomic reservation the total could reach 8 × $0.20.
    """

    async def go():
        enf = BudgetEnforcer(
            TaskBudget(max_steps=10_000, max_tokens=10**12, max_usd=1.0)
        )
        committed = 0.0
        lock = asyncio.Lock()

        async def consumer():
            nonlocal committed
            while True:
                try:
                    await enf.before_model_call(est_max_cost_usd=0.2)
                except BudgetExceeded:
                    return
                # simulate the model call happening
                await asyncio.sleep(0)
                await enf.record_model_call(
                    input_tokens=10, output_tokens=5, cost_usd=0.2, reserved_usd=0.2
                )
                async with lock:
                    committed += 0.2

        await asyncio.gather(*[consumer() for _ in range(8)])
        snap = await enf.snapshot()
        assert committed <= 1.0 + 1e-9, f"overshoot: committed={committed}"
        assert snap["usd"] <= 1.0 + 1e-9
        assert snap["usd_reserved"] == pytest.approx(0.0)

    asyncio.run(go())


def test_record_releases_reservation():
    async def go():
        enf = BudgetEnforcer(TaskBudget(max_usd=1.0))
        await enf.before_model_call(est_max_cost_usd=0.6)
        snap = await enf.snapshot()
        assert snap["usd_reserved"] == pytest.approx(0.6)
        await enf.record_model_call(
            input_tokens=1, output_tokens=1, cost_usd=0.1, reserved_usd=0.6
        )
        snap = await enf.snapshot()
        assert snap["usd_reserved"] == pytest.approx(0.0)
        assert snap["usd"] == pytest.approx(0.1)

    asyncio.run(go())


def test_stream_abort_records_partial():
    async def go():
        enf = BudgetEnforcer(TaskBudget())
        await enf.record_stream_abort(input_tokens=40, output_tokens=7, cost_usd=0.001)
        snap = await enf.snapshot()
        assert snap["tokens"] == 47
        assert snap["aborted_partials"] == 1

    asyncio.run(go())


async def test_daily_cap_enforced_at_gate(db, vault_env):
    """Dashboard daily caps are enforced, not just displayed: the ledger
    accrues every recorded call into KeyBudgetStore, and the enforcer's
    gate raises BudgetExceeded when spent_today + estimate > cap."""
    from byok_agent.ledger import Ledger
    from byok_agent.ops.budgets import KeyBudgetStore

    store = KeyBudgetStore(db)
    ledger = Ledger(db, key_budgets=store)
    await store.set_cap("k1", 1.0)
    await ledger.record(
        key_id="k1", provider="openai", model="openai/gpt-4o-mini",
        input_tokens=10, output_tokens=10, estimated_cost_usd=0.9,
    )
    enf = BudgetEnforcer(TaskBudget(max_usd=50.0), ledger=ledger, key_id="k1")
    with pytest.raises(BudgetExceeded):
        await enf.before_model_call(est_max_cost_usd=0.2)  # 0.9 + 0.2 > 1.0
    # Under the cap still passes.
    enf2 = BudgetEnforcer(TaskBudget(max_usd=50.0), ledger=ledger, key_id="k1")
    await enf2.before_model_call(est_max_cost_usd=0.05)  # 0.9 + 0.05 < 1.0


async def test_daily_cap_unlimited_without_row(db, vault_env):
    """No cap row => unlimited: the gate is a no-op, nothing breaks."""
    from byok_agent.ledger import Ledger
    from byok_agent.ops.budgets import KeyBudgetStore

    ledger = Ledger(db, key_budgets=KeyBudgetStore(db))
    enf = BudgetEnforcer(TaskBudget(max_usd=50.0), ledger=ledger, key_id="k9")
    await enf.before_model_call(est_max_cost_usd=5.0)
