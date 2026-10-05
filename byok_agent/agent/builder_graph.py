"""Phase 1 code builder: planner -> coders -> tester -> reviewer.

Graph::

    planner -> [Send] coder (x N, parallel, one per file)
            -> tester -> reviewer -> coder (on CHANGES, max 3 rounds)
                                   -> promote (on APPROVED or cap)

Every model call goes through the Phase 0
:class:`~byok_agent.agent.budgets.BudgetEnforcer` held in the graph
config — parallel coders share ONE enforcer, so its atomic counters bound
total spend (no N x overshoot).

Execution model:

- All file writes happen in the SANDBOX (a task container). The real
  project directory is never touched until the ``promote`` node.
- ``promote`` calls :func:`~langgraph.types.interrupt` with the full
  diff; a human (or ``auto_approve`` / an ``approver`` callback) decides.
  Only on approval are changed files copied sandbox -> real dir.
- Thread IDs stay ``user:{uid}:task:{tid}``; the Phase 0 checkpointer is
  reused. Builder runs are one-shot: re-running a *finished* tid returns
  the stored result idempotently; re-running an *unfinished* tid raises
  (the task container is gone after a crash — use a new tid).
"""
from __future__ import annotations

import json
import logging
import operator
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Send, interrupt

from .budgets import BudgetEnforcer, TaskBudget
from .graph import make_checkpointer, make_thread_id
from ..codemap import RepoMap
from ..edits import apply_edit, apply_whole_file, parse_edit_blocks, render_diff
from ..llm.router import RouteTarget

log = logging.getLogger(__name__)

MAX_REVIEW_ROUNDS = 3
MAX_CODER_MODEL_TOKENS = 2048


class ApprovalNeeded(Exception):
    """Raised when no approver/auto-approve is configured at the gate."""

    def __init__(self, diff: str, approval_id: int | None = None) -> None:
        super().__init__("promotion needs human approval")
        self.diff = diff
        # When an ApprovalStore is wired, the staged changes are persisted
        # under this id and the dashboard/CLI can approve out of band.
        self.approval_id = approval_id


class BuildError(Exception):
    """The build could not proceed (bad plan, unsafe path, ...)."""


class BuildInterruptedError(Exception):
    """An unfinished builder thread was re-run (sandbox is gone)."""


# -- prompts (role markers let tests route scripted model replies) --------

PLANNER_PROMPT = """[ROLE: PLANNER]
You are the PLANNER of a code-building agent. Break the user's task into
per-file work items.

Repository map:
{repomap}

Task: {task}

Output ONLY this JSON (no prose, no code fences):
{{"files": [{{"path": "relative/path.py", "instruction": "what to change in this file"}}], "test_command": "<shell command that verifies the change>"}}

Rules:
- "path" is repo-relative, never absolute, never containing "..".
- Keep the file list minimal: touch only what the task needs.
- "test_command" runs inside the task container at the repo root.
  Defaults: python -> "python -m pytest -q", node -> "npm test --silent",
  go -> "go test ./...", java -> "mvn -q -o test", rust -> "cargo test --offline --quiet".
"""

CODER_PROMPT = """[ROLE: CODER]
You are the CODER. Implement the file task below. Output ONLY edit blocks.

Repository map (context):
{repomap}

File: {path}
Current content of {path}:
<<<FILE
{content}
FILE>>>

Instruction: {instruction}
{feedback}

Edit block format (Aider-style), one or more:
<<<<<<< SEARCH
<exact text from the current file>
=======
<replacement text>
>>>>>>> REPLACE

For a NEW file, or a full rewrite of a file under 400 lines:
<<<<<<< WHOLE FILE
<entire new file content>
>>>>>>> END

Rules:
- SEARCH must match the current file content EXACTLY (copy it verbatim).
- One concern per block; multiple blocks allowed.
- Never output prose outside the blocks.
"""

REVIEWER_PROMPT = """[ROLE: REVIEWER]
You are the REVIEWER. Judge whether the change implements the task.

Task: {task}

Diff:
{diff}

Test output:
{test_output}

Output EXACTLY one of:

APPROVED

or

CHANGES:
<concrete per-file feedback for the coder>
"""


# -- state -----------------------------------------------------------------

class BuilderState(TypedDict):
    messages: Annotated[list, operator.add]
    uid: str
    tid: str
    project_dir: str
    task: str
    language: str
    test_command: str
    plan: list  # list[dict]: {"path":..., "instruction":...}
    file_task: dict  # set per-branch via Send
    review_feedback: str
    changes: Annotated[list, operator.add]  # [{"path":..., "round":...}]
    test_output: str
    review_rounds: int
    review_approved: bool  # reviewer's verdict on the diff
    approved: bool  # human promotion decision (set only in promote_node)
    pending_diff: str
    answer: str
    steps_used: int
    tokens_used: int
    usd_used: float


@dataclass
class BuildContext:
    """Wiring shared by all builder nodes (travels in graph config)."""

    router: Any
    ledger: Any
    primary: RouteTarget
    fallbacks: tuple = ()
    uid: str = ""
    auto_approve: bool = False
    approver: Any = None  # async (diff: str) -> bool


def _ctx_from_config(config: RunnableConfig) -> BuildContext:
    return config["configurable"]["_bctx"]


def _sandbox_from_config(config: RunnableConfig) -> Any:
    return config["configurable"]["_sandbox"]


def _repomap_from_config(config: RunnableConfig) -> str:
    return config["configurable"].get("_repomap", "")


def _enforcer_from_config(config: RunnableConfig) -> BudgetEnforcer:
    return config["configurable"]["_enforcer"]


async def _call_model(
    messages: list[dict[str, str]],
    *,
    ctx: BuildContext,
    config: RunnableConfig,
    max_tokens: int = MAX_CODER_MODEL_TOKENS,
) -> Any:
    """One budget-gated, metered model call (shared by all builder nodes)."""
    # Local import: byok_agent.ledger imports byok_agent.agent.budgets, so a
    # top-level import here would be circular (same pattern as graph.py).
    from ..ledger import estimate_cost, estimate_max_cost_usd

    enf = _enforcer_from_config(config)
    est_in = sum(len(m["content"]) // 4 for m in messages)
    est_cost = estimate_max_cost_usd(ctx.primary.model, est_in, max_tokens)
    await enf.before_model_call(
        est_input_tokens=est_in,
        est_max_output_tokens=max_tokens,
        est_max_cost_usd=est_cost,
    )

    async def attempt_sink(attempt: Any) -> None:
        await ctx.ledger.record(
            key_id=attempt.key_id,
            provider=attempt.provider,
            model=attempt.model,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            status=attempt.status,
            error=attempt.error,
        )

    try:
        result, target = await ctx.router.acomplete(
            messages,
            ctx.primary,
            ctx.fallbacks,
            owner=ctx.uid,
            attempt_sink=attempt_sink,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as exc:
        await enf.record_stream_abort(
            reserved_usd=est_cost,
            reason=f"builder model call failed: {type(exc).__name__}",
        )
        raise
    cost = estimate_cost(target.model, result.raw)
    await enf.record_model_call(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost,
        reserved_usd=est_cost,
    )
    await ctx.ledger.record(
        key_id=target.key_id,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_tokens=result.cached_tokens,
        estimated_cost_usd=cost,
    )
    return result


def _validate_relpath(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not str(path).strip():
        raise BuildError(f"unsafe plan path: {path!r}")
    return p.as_posix()


def _latest_changes(changes: list[dict]) -> list[str]:
    """Changed paths, deduplicated (a file edited in several rounds counts once)."""
    seen: list[str] = []
    for c in changes:
        p = c.get("path")
        if p and p not in seen:
            seen.append(p)
    return seen


# -- nodes -----------------------------------------------------------------

async def planner_node(state: BuilderState, config: RunnableConfig) -> dict:
    ctx = _ctx_from_config(config)
    repomap = _repomap_from_config(config)
    prompt = PLANNER_PROMPT.format(repomap=repomap, task=state["task"])
    result = await _call_model(
        [{"role": "user", "content": prompt}], ctx=ctx, config=config
    )
    m = re.search(r"\{.*\}", result.text, re.DOTALL)
    if not m:
        raise BuildError(f"planner did not emit JSON: {result.text[:200]!r}")
    try:
        plan_doc = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise BuildError(f"planner JSON unparseable: {exc}") from exc
    files = plan_doc.get("files") or []
    if not files:
        raise BuildError("planner produced an empty file list")
    plan = []
    for f in files:
        plan.append(
            {
                "path": _validate_relpath(f["path"]),
                "instruction": str(f.get("instruction", "")),
            }
        )
    test_command = str(plan_doc.get("test_command") or state.get("test_command") or "")
    log.info("plan: %d file(s), test=%r", len(plan), test_command)
    return {
        "messages": [{"role": "assistant", "content": f"PLAN: {len(plan)} file(s)"}],
        "plan": plan,
        "test_command": test_command,
    }


def fan_out_coders(state: BuilderState):
    return [
        Send("coder", {"file_task": ft, "review_feedback": ""}) for ft in state["plan"]
    ]


async def coder_node(state: BuilderState, config: RunnableConfig) -> dict:
    ctx = _ctx_from_config(config)
    sandbox = _sandbox_from_config(config)
    repomap = _repomap_from_config(config)
    ft = state.get("file_task") or {}
    path = _validate_relpath(ft.get("path", ""))
    instruction = ft.get("instruction", "")
    feedback = state.get("review_feedback", "")

    try:
        content = await sandbox.read_file(path)
    except Exception:
        content = ""  # new file

    feedback_block = (
        f"\nReviewer feedback from the previous round (address it):\n{feedback}\n"
        if feedback
        else ""
    )
    prompt = CODER_PROMPT.format(
        repomap=repomap, path=path, content=content, instruction=instruction,
        feedback=feedback_block,
    )
    result = await _call_model(
        [{"role": "user", "content": prompt}], ctx=ctx, config=config
    )
    blocks = parse_edit_blocks(result.text)
    if not blocks:
        raise BuildError(
            f"coder emitted no edit blocks for {path!r}: {result.text[:200]!r}"
        )

    new_content = content
    for b in blocks:
        if b.whole_file:
            new_content = apply_whole_file(new_content, b.replace)
        else:
            new_content = apply_edit(new_content, b.search, b.replace)
    await sandbox.write_file(path, new_content)
    log.info("coder wrote %s (%d block(s))", path, len(blocks))
    return {
        "messages": [{"role": "assistant", "content": f"EDITED {path}"}],
        "changes": [{"path": path, "round": state.get("review_rounds", 0)}],
    }


async def tester_node(state: BuilderState, config: RunnableConfig) -> dict:
    sandbox = _sandbox_from_config(config)
    enf = _enforcer_from_config(config)
    await enf.before_tool_call("sandbox_test")
    test_command = state.get("test_command") or ""
    if not test_command:
        return {"test_output": "(no test command configured; skipped)"}
    result = await sandbox.exec(test_command, timeout_s=300.0)
    output = (
        f"$ {test_command}\nexit={result.exit_code}\n"
        f"{result.stdout}\n{result.stderr}"
    ).strip()
    log.info("tester: exit=%d", result.exit_code)
    return {
        "messages": [{"role": "user", "content": f"TEST RESULT:\n{output}"}],
        "test_output": output,
    }


async def reviewer_node(state: BuilderState, config: RunnableConfig) -> dict:
    ctx = _ctx_from_config(config)
    sandbox = _sandbox_from_config(config)
    originals: dict[str, str] = config["configurable"].get("_originals", {})
    paths = _latest_changes(state.get("changes", []))
    diffs: list[str] = []
    for p in paths:
        before = originals.get(p, "")
        try:
            after = await sandbox.read_file(p)
        except Exception:
            after = ""
        if before != after:
            diffs.append(render_diff(before, after, p))
    diff = "\n".join(diffs) or "(no changes)"
    prompt = REVIEWER_PROMPT.format(
        task=state["task"], diff=diff, test_output=state.get("test_output", "")
    )
    result = await _call_model(
        [{"role": "user", "content": prompt}], ctx=ctx, config=config
    )
    text = result.text.strip()
    rounds = state.get("review_rounds", 0) + 1
    if text.upper().startswith("APPROVED"):
        log.info("reviewer: APPROVED (round %d)", rounds)
        return {
            "messages": [{"role": "assistant", "content": "REVIEW: APPROVED"}],
            "review_approved": True,
            "review_rounds": rounds,
            "pending_diff": diff,
        }
    feedback = text
    if feedback.upper().startswith("CHANGES:"):
        feedback = feedback[len("CHANGES:"):].strip()
    log.info("reviewer: CHANGES requested (round %d)", rounds)
    return {
        "messages": [{"role": "assistant", "content": f"REVIEW: CHANGES\n{feedback}"}],
        "review_approved": False,
        "review_feedback": feedback,
        "review_rounds": rounds,
        "pending_diff": diff,
    }


def route_after_review(state: BuilderState):
    if state.get("review_approved"):
        return "promote"
    if state.get("review_rounds", 0) >= MAX_REVIEW_ROUNDS:
        log.warning(
            "review cap (%d rounds) reached; sending to promote gate", MAX_REVIEW_ROUNDS
        )
        return "promote"
    return [
        Send(
            "coder",
            {"file_task": ft, "review_feedback": state.get("review_feedback", "")},
        )
        for ft in state["plan"]
    ]


async def promote_node(state: BuilderState, config: RunnableConfig) -> dict:
    """Human gate: sandbox -> real project dir ONLY on approval."""
    ctx = _ctx_from_config(config)
    sandbox = _sandbox_from_config(config)
    diff = state.get("pending_diff", "(no diff)")
    paths = _latest_changes(state.get("changes", []))

    if ctx.auto_approve:
        approved = True
    else:
        # Suspends here; run() resumes with Command(resume=bool).
        approved = interrupt(
            {
                "question": "Apply these changes to the REAL project directory?",
                "diff": diff,
                "paths": paths,
            }
        )
        approved = bool(approved)

    if approved:
        dest = Path(state["project_dir"])
        await sandbox.copy_out(dest, paths=paths or None)
        log.info("promoted %d file(s) to %s", len(paths), dest)
        answer = f"Build complete: {len(paths)} file(s) updated ({', '.join(paths)})."
    else:
        log.info("promotion rejected by approver; sandbox changes discarded")
        answer = "Build rejected at the approval gate; no files were changed."
    enf = _enforcer_from_config(config)
    snap = await enf.snapshot()
    return {
        "approved": approved,
        "answer": answer,
        "steps_used": snap["steps"],
        "tokens_used": snap["tokens"],
        "usd_used": snap["usd"],
    }


# -- runner ------------------------------------------------------------------

class BuilderRunner:
    """Runs the planner -> coder -> tester -> reviewer graph for one task."""

    def __init__(
        self,
        *,
        router: Any,
        ledger: Any,
        make_enforcer: Any,
        primary: RouteTarget,
        fallbacks: tuple = (),
        checkpointer: Any = None,
        sandbox_factory: Any = None,
        auto_approve: bool = False,
        approver: Any = None,
        approval_store: Any = None,
        staging_root: str | Path | None = None,
    ) -> None:
        self._router = router
        self._ledger = ledger
        self._make_enforcer = make_enforcer
        self._primary = primary
        self._fallbacks = fallbacks
        self._checkpointer = checkpointer
        self._sandbox_factory = sandbox_factory
        self._auto_approve = auto_approve
        self._approver = approver
        # Phase 4: when set and the promote gate suspends with no approver,
        # changed files are staged on the host and an approval record is
        # persisted so the dashboard/CLI can decide out of band.
        self._approval_store = approval_store
        self._staging_root = (
            Path(staging_root) if staging_root is not None
            else Path(os.environ.get("BYOK_STAGING_ROOT", "./approvals_staging"))
        )

    def _build(self):
        from langgraph.graph import END, StateGraph

        builder = StateGraph(BuilderState)
        builder.add_node("planner", planner_node)
        builder.add_node("coder", coder_node)
        builder.add_node("tester", tester_node)
        builder.add_node("reviewer", reviewer_node)
        builder.add_node("promote", promote_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges("planner", fan_out_coders)
        builder.add_edge("coder", "tester")
        builder.add_edge("tester", "reviewer")
        builder.add_conditional_edges(
            "reviewer", route_after_review, {"promote": "promote"}
        )
        builder.add_edge("promote", END)
        return builder.compile(checkpointer=self._checkpointer)

    def _snapshot_originals(self, project_dir: Path) -> dict[str, str]:
        """Read every text file under the project dir (for diffing)."""
        originals: dict[str, str] = {}
        for p in project_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(project_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                if p.stat().st_size > 500_000:
                    continue
                originals[rel.as_posix()] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return originals

    async def run(
        self,
        *,
        uid: str,
        tid: str,
        project_dir: str | Path,
        task_instruction: str,
        language: str = "python",
        test_command: str | None = None,
        budget: TaskBudget | None = None,
    ) -> dict:
        """Build: plan -> code (sandbox) -> test -> review -> approve -> promote.

        Raises :class:`ApprovalNeeded` if the promote gate suspends and no
        approver/auto-approve is configured.
        """
        from ..sandbox.docker import DEFAULT_TEST_COMMANDS, DockerSandbox

        thread_id = make_thread_id(uid, tid)
        real_dir = Path(project_dir).resolve()
        if not real_dir.is_dir():
            raise BuildError(f"project dir not found: {real_dir}")

        graph = self._build()
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 60,
        }

        # One-shot semantics: finished tid -> idempotent re-read;
        # unfinished tid -> refuse (the task container is gone).
        existing = await graph.aget_state(config)
        values = existing.values or {}
        if values:
            if values.get("answer"):
                log.info("thread %s already finished; returning stored result", thread_id)
                return dict(values)
            raise BuildInterruptedError(
                f"thread {thread_id} is unfinished and its sandbox is gone; "
                "use a new task id"
            )

        enforcer = self._make_enforcer(budget or TaskBudget())
        originals = self._snapshot_originals(real_dir)
        repomap = RepoMap().build(real_dir)

        factory = self._sandbox_factory
        if factory is None:
            def factory() -> Any:  # noqa: F811
                return DockerSandbox(language=language, enforcer=enforcer)

        sandbox = factory()
        # Inject the enforcer when the factory didn't (tests often build the
        # sandbox first, then the runner).
        if getattr(sandbox, "_enforcer", None) is None and hasattr(sandbox, "_enforcer"):
            sandbox._enforcer = enforcer

        ctx = BuildContext(
            router=self._router,
            ledger=self._ledger,
            primary=self._primary,
            fallbacks=self._fallbacks,
            uid=uid,
            auto_approve=self._auto_approve,
            approver=self._approver,
        )
        config["configurable"].update(
            {
                "_enforcer": enforcer,
                "_sandbox": sandbox,
                "_bctx": ctx,
                "_repomap": repomap,
                "_originals": originals,
            }
        )

        initial: BuilderState = {
            "messages": [{"role": "user", "content": task_instruction}],
            "uid": uid,
            "tid": tid,
            "project_dir": str(real_dir),
            "task": task_instruction,
            "language": language,
            "test_command": test_command
            or DEFAULT_TEST_COMMANDS.get(language, ""),
            "plan": [],
            "file_task": {},
            "review_feedback": "",
            "changes": [],
            "test_output": "",
            "review_rounds": 0,
            "review_approved": False,
            "approved": False,
            "pending_diff": "",
            "answer": "",
            "steps_used": 0,
            "tokens_used": 0,
            "usd_used": 0.0,
        }

        await sandbox.start()
        try:
            await sandbox.copy_in(real_dir)
            # langgraph >= 0.6: ainvoke does NOT raise on interrupt() — the
            # pending interrupt is returned in result["__interrupt__"].
            # (Older versions raised GraphInterrupt; we handle both.)
            try:
                result = await graph.ainvoke(initial, config)
            except GraphInterrupt:
                result = {"__interrupt__": [True]}
            if result.get("__interrupt__"):
                st = await graph.aget_state(config)
                diff = (st.values or {}).get("pending_diff", "")
                if self._auto_approve:
                    approved = True
                elif self._approver is not None:
                    approved = await self._approver(diff)
                elif self._approval_store is not None:
                    # Phase 4: persist the gate. Stage the changed files on
                    # the host BEFORE the sandbox is torn down, then record
                    # an approval the dashboard/CLI can decide out of band.
                    approval_id = await self._stage_approval(
                        sandbox, st.values or {}, thread_id, uid, diff
                    )
                    raise ApprovalNeeded(diff, approval_id=approval_id)
                else:
                    raise ApprovalNeeded(diff)
                result = await graph.ainvoke(Command(resume=bool(approved)), config)
            result = dict(result)
            result.pop("__interrupt__", None)
            result["budget"] = await enforcer.snapshot()
            result["thread_id"] = thread_id
            return result
        finally:
            await sandbox.stop()

    async def _stage_approval(
        self,
        sandbox: Any,
        values: dict,
        thread_id: str,
        uid: str,
        diff: str,
    ) -> int:
        """Copy changed files out of the sandbox and persist an approval.

        Returns the approval id. The staging dir lives on the HOST (never in
        the sandbox) and is removed when the approval is decided.
        """
        import shutil

        paths = _latest_changes(values.get("changes", []))
        staging = self._staging_root / thread_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        # copy_out(local_dir, paths) pulls the changed files to the host.
        await sandbox.copy_out(staging, paths or None)
        summary = f"build {thread_id}: {len(paths)} file(s) changed"
        payload = {
            "thread_id": thread_id,
            "owner": uid,
            "project_dir": values.get("project_dir", ""),
            "paths": paths,
            "staging_dir": str(staging),
            "diff_preview": diff[:2000],
        }
        store = self._approval_store
        return await store.request(
            thread_id=thread_id,
            kind="build-promote",
            summary=summary,
            payload=payload,
        )


async def apply_approval(
    db: Any, approval_id: int, *, approved: bool, decided_by: str
) -> dict:
    """Decide a persisted build-promote approval.

    On approve: staged files are copied into the real project directory
    (path-traversal guarded). On deny: nothing is applied. Either way the
    staging dir is removed and the approval is marked decided.

    Returns {"applied": [paths]} on approve, {"applied": []} on deny.
    """
    import shutil

    from ..ops.approvals import ApprovalStore

    store = ApprovalStore(db)
    rec = await store.get(approval_id)
    if rec is None:
        raise KeyError(f"approval {approval_id} not found")
    if rec.status != "pending":
        raise ValueError(f"approval {approval_id} already {rec.status}")
    if rec.kind != "build-promote":
        # Non-build approvals (e.g. deploy gates) are decided without files.
        await store.decide(approval_id, approved=approved, decided_by=decided_by)
        return {"applied": [], "kind": rec.kind}

    payload = rec.payload or {}
    staging = Path(payload.get("staging_dir", ""))
    project_dir = Path(payload.get("project_dir", ""))
    applied: list[str] = []
    try:
        if approved:
            if not project_dir.is_dir():
                raise ValueError(f"project dir gone: {project_dir}")
            for rel in payload.get("paths", []):
                # Guard: repo-relative only, no "..", no absolute.
                if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                    raise ValueError(f"unsafe staged path: {rel!r}")
                src = staging / rel
                if not src.is_file():
                    continue
                dest = project_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                applied.append(rel)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    await store.decide(approval_id, approved=approved, decided_by=decided_by)
    return {"applied": applied}
