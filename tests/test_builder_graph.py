"""Phase 1 tests: planner -> coder -> tester -> reviewer graph.

Model + vault + ledger are real (SQLite); the LLM client is scripted by
[ROLE: ...] prompt markers; the sandbox is FakeSandbox. No network.
"""
from __future__ import annotations

import json
import re

import pytest

from byok_agent.agent import TaskBudget, make_checkpointer
from byok_agent.agent.budgets import BudgetEnforcer
from byok_agent.agent.builder_graph import BuilderRunner
from byok_agent.llm import RouteTarget, Router
from byok_agent.llm.client import CompletionResult
from byok_agent.sandbox import FakeSandbox, SandboxResult


class ScriptedClient:
    """Canned replies routed by the [ROLE: X] marker in the prompt."""

    def __init__(self, reviewer_script: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self._reviewer_script = list(reviewer_script or ["APPROVED"])

    def role_calls(self, role: str) -> int:
        return sum(1 for c in self.calls if f"[ROLE: {role}]" in c)

    async def acomplete(
        self, messages, model, *, api_key, max_tokens=1024,
        temperature=0.2, **kwargs,
    ) -> CompletionResult:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "[ROLE: PLANNER]" in prompt:
            text = json.dumps(
                {
                    "files": [
                        {"path": "app.py", "instruction": "add a health() function"},
                        {"path": "util.py", "instruction": "add a slugify() helper"},
                    ],
                    "test_command": "echo TESTS-OK",
                }
            )
        elif "[ROLE: CODER]" in prompt:
            m = re.search(r"^File: (\S+)", prompt, re.M)
            path = m.group(1) if m else ""
            if path == "app.py":
                text = (
                    '<<<<<<< SEARCH\nprint("orig")\n=======\nprint("orig")\n\n'
                    "\ndef health():\n    return True\n>>>>>>> REPLACE"
                )
            else:
                text = (
                    "<<<<<<< WHOLE FILE\ndef slugify(t):\n"
                    '    return t.lower().replace(" ", "-")\n>>>>>>> END'
                )
        elif "[ROLE: REVIEWER]" in prompt:
            text = (
                self._reviewer_script.pop(0)
                if self._reviewer_script
                else "APPROVED"
            )
        else:
            text = "ANSWER: done"
        return CompletionResult(
            text=text, model=model, provider="openai",
            input_tokens=10, output_tokens=20,
        )


@pytest.fixture()
def proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "app.py").write_text('print("orig")\n')
    return d


async def _make_runner(vault, ledger, client, proj, tmp_path,
                       reviewer_script=None, auto_approve=True,
                       approver=None, task_budget=None):
    record = await vault.add_key(
        label="t", provider="openai", plaintext_key="sk-test-not-real", owner="u1"
    )
    router = Router(client, vault)
    primary = RouteTarget(key_id=record.key_id, model="openai/gpt-4o-mini")

    def make_enforcer(b: TaskBudget) -> BudgetEnforcer:
        return BudgetEnforcer(
            b, ledger=ledger, key_id=record.key_id, key_max_usd=None
        )

    handlers = {"echo TESTS-OK": SandboxResult(0, "TESTS-OK\n", "")}

    def factory():
        return FakeSandbox(handlers=handlers, workdir=tmp_path / "sbx")

    async with make_checkpointer(None) as checkpointer:
        runner = BuilderRunner(
            router=router,
            ledger=ledger,
            make_enforcer=make_enforcer,
            primary=primary,
            checkpointer=checkpointer,
            sandbox_factory=factory,
            auto_approve=auto_approve,
            approver=approver,
        )
        result = await runner.run(
            uid="u1", tid="t1", project_dir=proj,
            task_instruction="add health endpoint and slugify helper",
            language="python",
            budget=task_budget or TaskBudget(max_steps=40, max_tokens=10**6,
                                            max_usd=100.0),
        )
    return result, client


async def test_full_build_reject_then_approve(vault, ledger, proj, tmp_path):
    client = ScriptedClient(reviewer_script=["CHANGES:\n- app.py: tidy up", "APPROVED"])
    result, _ = await _make_runner(vault, ledger, client, proj, tmp_path,
                                   reviewer_script=["CHANGES:\n- app.py: tidy up",
                                                    "APPROVED"])
    assert result["approved"] is True
    assert result["review_rounds"] == 2
    assert client.role_calls("REVIEWER") == 2
    # Both files promoted to the REAL project dir.
    app = (proj / "app.py").read_text()
    assert "def health():" in app
    assert "def slugify" in (proj / "util.py").read_text()
    # Shared budget: 1 planner + 2x2 coders + 2 reviewers = 7 model calls.
    assert result["budget"]["steps"] == 7
    assert client.role_calls("PLANNER") == 1
    assert client.role_calls("CODER") == 4


async def test_review_cap_stops_loop(vault, ledger, proj, tmp_path):
    client = ScriptedClient(reviewer_script=["CHANGES: again"] * 10)
    result, _ = await _make_runner(vault, ledger, client, proj, tmp_path,
                                   reviewer_script=["CHANGES: again"] * 10)
    assert result["review_rounds"] == 3  # capped, no infinite loop
    assert client.role_calls("REVIEWER") == 3
    assert result["approved"] is True  # cap -> promote gate -> auto-approved


async def test_rejection_discards_changes(vault, ledger, proj, tmp_path):
    async def never_approve(diff: str) -> bool:
        assert "health" in diff
        return False

    client = ScriptedClient()
    result, _ = await _make_runner(vault, ledger, client, proj, tmp_path,
                                   auto_approve=False, approver=never_approve)
    assert result["approved"] is False
    assert "rejected" in result["answer"].lower()
    # Real dir untouched.
    assert (proj / "app.py").read_text() == 'print("orig")\n'
    assert not (proj / "util.py").exists()


async def test_finished_tid_is_idempotent(vault, ledger, proj, tmp_path, capsys):
    client = ScriptedClient()
    record = await vault.add_key(
        label="t", provider="openai", plaintext_key="sk-test-not-real", owner="u1"
    )
    router = Router(client, vault)
    primary = RouteTarget(key_id=record.key_id, model="openai/gpt-4o-mini")

    def make_enforcer(b):
        return BudgetEnforcer(b, ledger=ledger, key_id=record.key_id)

    def factory():
        return FakeSandbox(
            handlers={"echo TESTS-OK": SandboxResult(0, "TESTS-OK\n", "")},
            workdir=tmp_path / "sbx",
        )

    async with make_checkpointer(None) as checkpointer:
        runner = BuilderRunner(
            router=router, ledger=ledger, make_enforcer=make_enforcer,
            primary=primary, checkpointer=checkpointer,
            sandbox_factory=factory, auto_approve=True,
        )
        r1 = await runner.run(uid="u1", tid="t9", project_dir=proj,
                              task_instruction="do it", language="python")
        calls_after_first = len(client.calls)
        r2 = await runner.run(uid="u1", tid="t9", project_dir=proj,
                              task_instruction="do it", language="python")
    assert r1["answer"] == r2["answer"]
    assert len(client.calls) == calls_after_first  # zero new model calls
