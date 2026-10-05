"""Phase 0 agent: a manual ReAct loop on a LangGraph StateGraph.

Nodes: ``agent`` (one model call) -> ``tools`` (execute actions) -> ``agent``
until the model emits ``ANSWER:`` or the iteration cap (25) is hit.

Tool protocol (text-based ReAct — transparent and provider-agnostic)::

    THOUGHT: <reasoning>
    ACTION: <tool_name>
    ARGS: {"path": "notes/todo.txt"}

    ... repeat ...

    ANSWER: <final response to the user>

Phase 1 may switch to native function-calling; the budget gates and the
ledger recording stay identical.

Thread IDs are ALWAYS ``user:{uid}:task:{tid}`` — never a bare
``user:{uid}`` — so concurrent tasks never collide and crash-resume
targets the right run.

Checkpointing: an ASYNC graph requires the async saver. We use
``AsyncPostgresSaver`` when a Postgres DSN is configured; otherwise we
fall back to the in-memory saver with a LOUD warning (no persistence
across restarts in dev).

The :class:`BudgetEnforcer` is per-run runtime state: it travels in the
LangGraph *config* (``config["configurable"]["_enforcer"]``), never in
checkpointed graph state. On resume it is rebuilt from the persisted
``steps_used`` / ``tokens_used`` / ``usd_used`` mirrors via
:meth:`BudgetEnforcer.restore`.

A model reply with no ACTION block and no ``ANSWER:`` prefix is accepted as
a plain-text final answer (the ``ANSWER:`` prefix stays preferred, not
required).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, TypedDict

from langchain_core.runnables import RunnableConfig

log = logging.getLogger(__name__)

MAX_ITERATIONS = 25

_THREAD_RE = re.compile(r"^user:[^:\s/]+:task:[^:\s/]+$")

SYSTEM_PROMPT = """You are a careful, thrifty assistant. You have tools; use them only when needed.

To use a tool, output EXACTLY this block (no code fences):

THOUGHT: <one or two sentences of reasoning>
ACTION: <tool_name>
ARGS: <JSON object of arguments>

After the tool result comes back as OBSERVATION, continue reasoning. When you
are done, output EXACTLY:

ANSWER: <your final response to the user>

Rules:
- One ACTION per turn. Never invent tool names or arguments.
- Prefer ANSWER over more tool calls when you have enough information.
- Keep ARGS valid JSON.

Available tools:
{tools}
"""


def make_thread_id(uid: str, tid: str) -> str:
    """Build a validated ``user:{uid}:task:{tid}`` thread id."""
    for name, value in (("uid", uid), ("tid", tid)):
        if not value or not str(value).strip():
            raise ValueError(f"{name} must be a non-empty string")
        if re.search(r"[:\s/]", str(value)):
            raise ValueError(f"{name} must not contain ':', whitespace or '/'")
    thread_id = f"user:{uid}:task:{tid}"
    return validate_thread_id(thread_id)


def validate_thread_id(thread_id: str) -> str:
    """Reject anything that is not ``user:{uid}:task:{tid}``."""
    if not _THREAD_RE.match(thread_id or ""):
        raise ValueError(
            f"invalid thread id {thread_id!r}: must match 'user:<uid>:task:<tid>'"
        )
    return thread_id


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a THOUGHT/ACTION/ARGS block. Returns (name, args) or None."""
    action_m = re.search(r"^ACTION:\s*(\w+)\s*$", text, re.MULTILINE)
    if not action_m:
        return None
    args_m = re.search(r"^ARGS:\s*(\{.*\})\s*$", text, re.MULTILINE | re.DOTALL)
    args: dict[str, Any] = {}
    if args_m:
        try:
            args = json.loads(args_m.group(1))
        except json.JSONDecodeError:
            return (action_m.group(1), {"_raw": args_m.group(1)})
    return action_m.group(1), args


def extract_answer(text: str) -> str | None:
    """Return the ANSWER: payload if the model finished, else None."""
    m = re.search(r"^ANSWER:\s*(.*)$", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else None


def _concat_messages(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    """Checkpointed graph state (budget mirrors are observability only)."""

    messages: Annotated[list, _concat_messages]
    uid: str
    tid: str
    key_id: str
    model: str
    iterations: int
    answer: str
    steps_used: int
    tokens_used: int
    usd_used: float


def _enforcer_from_config(config: RunnableConfig) -> Any:
    return config["configurable"]["_enforcer"]


@asynccontextmanager
async def make_checkpointer(
    postgres_dsn: str | None,
) -> AsyncIterator[Any]:
    """Yield a checkpointer for the async graph.

    - With a Postgres DSN: ``AsyncPostgresSaver`` (persistent; the saver and
      its connection live for the whole ``async with`` block).
    - Otherwise: in-memory saver with a LOUD warning (dev only — no
      crash-resume across restarts).

    Usage::

        async with make_checkpointer(dsn) as checkpointer:
            runner = AgentRunner(..., checkpointer=checkpointer)
            await runner.run(...)
    """
    saver_cm = None
    saver = None
    if postgres_dsn:
        try:
            # v2.x ships as the namespace subpackage
            # langgraph.checkpoint.postgres.aio (the top-level
            # langgraph_checkpoint_postgres module does not exist).
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # from_conn_string is an async CONTEXT MANAGER in v2.x.
            saver_cm = AsyncPostgresSaver.from_conn_string(postgres_dsn)
            saver = await saver_cm.__aenter__()
            await saver.setup()  # creates checkpoint tables
            log.info("checkpointer: AsyncPostgresSaver (persistent)")
        except Exception as exc:
            # NEVER degrade silently: losing the checkpointer loses
            # crash-resume, and a quiet warning is how that goes unnoticed.
            log.error("Postgres checkpointer unavailable: %s", exc)
            print(
                "ERROR: Postgres checkpointer unavailable "
                f"({exc}).\n"
                "  -> This is usually a missing libpq: install "
                "'psycopg[binary]' (see requirements.txt).\n"
                "  -> Falling back to the IN-MEMORY saver: runs will NOT "
                "survive restarts and resume is dev-only.",
                file=sys.stderr,
            )
            saver = None
    if saver is None:
        print(
            "WARNING: using in-memory LangGraph checkpointer — "
            "no crash-resume across restarts. Set DATABASE_URL for Postgres."
        )
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
    else:
        try:
            yield saver
        finally:
            await saver_cm.__aexit__(None, None, None)


class AgentRunner:
    """Wires vault -> router -> ledger -> budgets -> graph for one task."""

    def __init__(
        self,
        *,
        router: Any,
        ledger: Any,
        make_enforcer: Any,
        primary: Any,
        fallbacks: tuple = (),
        checkpointer: Any = None,
    ) -> None:
        self._router = router
        self._ledger = ledger
        self._make_enforcer = make_enforcer
        self._primary = primary
        self._fallbacks = fallbacks
        self._checkpointer = checkpointer

    # -- graph construction (enforcer rides in config, not in the builder) --
    def _build(self):
        from langgraph.graph import END, StateGraph

        from . import tools as toolmod
        from ..ledger import estimate_cost, estimate_max_cost_usd

        runner = self

        async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
            enf = _enforcer_from_config(config)
            history = list(state["messages"])
            prompt = SYSTEM_PROMPT.format(tools=toolmod.describe_tools())
            chat = [{"role": "system", "content": prompt}]
            for m in history:
                chat.append({"role": m["role"], "content": m["content"]})

            async def attempt_sink(attempt: Any) -> None:
                # Meter EVERY attempt — failed and skipped fallbacks included,
                # not just the call that eventually succeeded.
                await runner._ledger.record(
                    key_id=attempt.key_id,
                    provider=attempt.provider,
                    model=attempt.model,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
                    status=attempt.status,
                    error=attempt.error,
                )

            est_tokens = sum(len(m["content"]) // 4 for m in chat)
            est_cost = estimate_max_cost_usd(
                runner._primary.model, est_tokens, max_output_tokens=1024
            )
            await enf.before_model_call(
                est_input_tokens=est_tokens,
                est_max_output_tokens=1024,
                est_max_cost_usd=est_cost,
            )

            try:
                result, target = await runner._router.acomplete(
                    chat,
                    runner._primary,
                    runner._fallbacks,
                    owner=state["uid"],
                    attempt_sink=attempt_sink,
                    max_tokens=1024,
                    temperature=0.2,
                )
            except Exception as exc:
                # Transport failure / stream abort: record what we know so
                # the cost is not silently under-recorded, and RELEASE the
                # reservation — a failed call must not pin headroom forever.
                await enf.record_stream_abort(
                    reserved_usd=est_cost,
                    reason=f"model call failed: {type(exc).__name__}",
                )
                raise

            cost = estimate_cost(target.model, result.raw)
            await enf.record_model_call(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=cost,
                reserved_usd=est_cost,
            )
            await runner._ledger.record(
                key_id=target.key_id,
                provider=result.provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
                estimated_cost_usd=cost,
            )
            snap = await enf.snapshot()
            answer = extract_answer(result.text)
            if not answer and parse_action(result.text) is None:
                # Plain-text final answer: the model finished without the
                # ANSWER: prefix and without a tool call. Accept it rather
                # than discarding the response.
                answer = result.text.strip()
            return {
                "messages": [{"role": "assistant", "content": result.text}],
                "iterations": state["iterations"] + 1,
                "answer": answer or "",
                "steps_used": snap["steps"],
                "tokens_used": snap["tokens"],
                "usd_used": snap["usd"],
            }

        async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
            enf = _enforcer_from_config(config)
            last = state["messages"][-1]["content"]
            parsed = parse_action(last)
            if parsed is None:
                observation = (
                    "ERROR: no ACTION block found. Either call a tool with "
                    "THOUGHT/ACTION/ARGS or finish with ANSWER:"
                )
            else:
                name, args = parsed
                await enf.before_tool_call(name)
                if isinstance(args, dict) and "_raw" in args:
                    observation = f"ERROR: ARGS is not valid JSON: {args['_raw']!r}"
                else:
                    observation = toolmod.run_tool(name, args)
            return {
                "messages": [{"role": "user", "content": f"OBSERVATION:\n{observation}"}],
            }

        def route_after_agent(state: AgentState) -> str:
            if state.get("answer"):
                return "end"
            if state["iterations"] >= MAX_ITERATIONS:
                return "end"
            last = state["messages"][-1]["content"]
            return "tools" if parse_action(last) else "end"

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent", route_after_agent, {"tools": "tools", "end": END}
        )
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=self._checkpointer)

    # -- public API ----------------------------------------------------------
    async def run(
        self,
        *,
        uid: str,
        tid: str,
        user_message: str,
        budget: Any = None,
    ) -> dict:
        """Run one task — or RESUME it if the thread already exists.

        Resume semantics (``user:{uid}:task:{tid}`` is the task identity):

        - No checkpoint for this thread → fresh run.
        - Checkpoint exists and the task is NOT finished (no answer yet) →
          resume with ``ainvoke(None, config)``: history continues, the
          iteration counter continues, and the budget counters are restored
          from the persisted ``steps_used`` / ``tokens_used`` / ``usd_used``
          mirrors. A task that hit ``max_steps=1`` stays finished — it does
          NOT get a free second run.
        - Checkpoint exists and the task IS finished → return the stored
          result idempotently (same tid = same task; a new task needs a new
          tid).

        Returns final state incl. ``answer`` and ``budget``.
        """
        from .budgets import TaskBudget

        thread_id = make_thread_id(uid, tid)
        graph = self._build()
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_ITERATIONS * 2 + 5,
        }

        existing = await graph.aget_state(config)
        values = existing.values or {}

        if values and not values.get("answer"):
            # INTERRUPTED task: resume from the checkpoint.
            enforcer = self._make_enforcer(budget or TaskBudget())
            await enforcer.restore(
                steps=int(values.get("steps_used", 0) or 0),
                tokens=int(values.get("tokens_used", 0) or 0),
                usd=float(values.get("usd_used", 0.0) or 0.0),
            )
            config["configurable"]["_enforcer"] = enforcer
            log.info(
                "resuming thread %s at iteration %s (budget restored: %s)",
                thread_id,
                values.get("iterations", 0),
                await enforcer.snapshot(),
            )
            result = await graph.ainvoke(None, config)
            result["budget"] = await enforcer.snapshot()
            result["thread_id"] = thread_id
            return result

        if values and values.get("answer"):
            # FINISHED task: idempotent re-read, no new work, no new spend.
            enforcer = self._make_enforcer(budget or TaskBudget())
            await enforcer.restore(
                steps=int(values.get("steps_used", 0) or 0),
                tokens=int(values.get("tokens_used", 0) or 0),
                usd=float(values.get("usd_used", 0.0) or 0.0),
            )
            result = dict(values)
            result["budget"] = await enforcer.snapshot()
            result["thread_id"] = thread_id
            log.info("thread %s already finished; returning stored result", thread_id)
            return result

        # FRESH task.
        enforcer = self._make_enforcer(budget or TaskBudget())
        config["configurable"]["_enforcer"] = enforcer
        initial: AgentState = {
            "messages": [{"role": "user", "content": user_message}],
            "uid": uid,
            "tid": tid,
            "key_id": self._primary.key_id,
            "model": self._primary.model,
            "iterations": 0,
            "answer": "",
            "steps_used": 0,
            "tokens_used": 0,
            "usd_used": 0.0,
        }
        result = await graph.ainvoke(initial, config)
        result["budget"] = await enforcer.snapshot()
        result["thread_id"] = thread_id
        return result
