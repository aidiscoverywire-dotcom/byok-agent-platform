"""End-to-end ReAct loop test with a fake model — no network, no API keys.

Guards the LangGraph wiring: node signatures (RunnableConfig injection),
tool execution, budget gating, ledger recording, and thread-id format.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from byok_agent.agent import TaskBudget, make_checkpointer
from byok_agent.agent.budgets import BudgetEnforcer
from byok_agent.agent.graph import AgentRunner
from byok_agent.llm import CompletionResult, RouteTarget, Router, provider_of
from byok_agent.vault import KeyRecord, KeyStatus


class FakeClient:
    def __init__(self):
        self.n = 0

    async def acomplete(self, messages, model, *, api_key, **kwargs):
        self.n += 1
        assert api_key == "sk-fake"
        if self.n == 1:
            text = "THOUGHT: check the time.\nACTION: get_current_time\nARGS: {}\n"
        else:
            text = "ANSWER: all done."
        return CompletionResult(
            text=text,
            model=model,
            provider=provider_of(model),
            input_tokens=50,
            output_tokens=20,
        )


class FakeVault:
    async def get_key(self, key_id: str, owner: str | None = None) -> str:
        return "sk-fake"

    async def get_record(self, key_id: str, owner: str | None = None) -> KeyRecord:
        return KeyRecord(
            key_id=key_id,
            label="t",
            provider="openai",
            encrypted_key=b"x",
            status=KeyStatus.ACTIVE,
        )


class FakeLedger:
    def __init__(self):
        self.rows: list[dict] = []

    async def record(self, **kw) -> None:
        self.rows.append(kw)

    async def get_spend(self, key_id: str) -> float:
        return 0.0

    async def reserve_key(self, key_id: str, est_usd: float, cap_usd: float) -> None:
        return None

    async def release_key_reservation(self, key_id: str, est_usd: float) -> None:
        return None


async def test_react_loop_runs_tool_then_answers():
    ledger = FakeLedger()
    async with make_checkpointer(None) as cp:  # in-memory + warning
        client = FakeClient()
        runner = AgentRunner(
            router=Router(client, FakeVault()),
            ledger=ledger,
            make_enforcer=lambda b: BudgetEnforcer(b),
            primary=RouteTarget("k1", "openai/gpt-4o-mini"),
            checkpointer=cp,
        )
        res = await runner.run(
            uid="u1",
            tid="t1",
            user_message="what time is it?",
            budget=TaskBudget(max_steps=5, max_usd=1.0),
        )
    assert res["thread_id"] == "user:u1:task:t1"
    assert res["answer"] == "all done."
    assert client.n == 2  # one tool call, then the answer
    assert len(ledger.rows) == 2  # every model call metered
    assert all(r["key_id"] == "k1" for r in ledger.rows)
    assert res["budget"]["steps"] == 2
