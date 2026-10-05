"""Phase 0 toolset — deliberately tiny and safe.

Three tools only: a clock, a workspace-scoped file reader, and a search
stub. There is NO shell execution in Phase 0 (that arrives with the
sandboxed executor in Phase 1).

``read_file`` is confined to ``AGENT_WORKSPACE`` (default ``./workspace``):
the requested path is resolved and must stay inside the root — ``..``
traversal, absolute escapes, and symlinks pointing out are rejected.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MAX_READ_BYTES = 100_000  # 100 KB cap per read


def _workspace_root() -> Path:
    return Path(os.environ.get("AGENT_WORKSPACE", "./workspace")).resolve()


def get_current_time() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def read_file(path: str) -> str:
    """Read a text file inside the agent workspace.

    Returns the file content, or an ``ERROR: ...`` string if the path is
    outside the workspace, unreadable, or too large.
    """
    root = _workspace_root()
    try:
        target = (root / path).resolve()
    except Exception:
        return "ERROR: invalid path"
    # Resolve symlinks too: target must still be inside root.
    try:
        target.relative_to(root)
    except ValueError:
        return "ERROR: path escapes the workspace"
    if not target.is_file():
        return f"ERROR: not a file: {path!r}"
    try:
        if target.stat().st_size > MAX_READ_BYTES:
            return f"ERROR: file too large (> {MAX_READ_BYTES} bytes)"
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: cannot read file: {exc}"


def web_search_stub(query: str) -> str:
    """Placeholder for web search (wired in Phase 3 with Tavily/Exa)."""
    return (
        "Web search is not wired in Phase 0 — it lands in Phase 3 "
        f"(Tavily/Exa). Your query was: {query!r}. "
        "Answer from your own knowledge and say so."
    )


ToolFn = Callable[..., str]

#: name -> (callable, one-line description for the model)
# Optional process-wide guard hook (see byok_agent.channels.guard).
# None by default: run_tool behavior is then exactly as before.
_tool_guard: Any = None

TOOLS: dict[str, tuple[ToolFn, str]] = {    "get_current_time": (
        get_current_time,
        "Current UTC time. Args: none.",
    ),
    "read_file": (
        read_file,
        "Read a text file inside the agent workspace. Args: {\"path\": \"notes/todo.txt\"}.",
    ),
    "web_search_stub": (
        web_search_stub,
        "Web search placeholder (Phase 0 returns a notice). Args: {\"query\": \"...\"}.",
    ),
}


def describe_tools() -> str:
    """Render the tool catalogue for the system prompt."""
    lines = [f"- {name}: {desc}" for name, (_, desc) in TOOLS.items()]
    return "\n".join(lines)


def run_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name. Unknown tools return an error string.

    When a :class:`~byok_agent.channels.guard.ToolGuard` is installed via
    ``install_tool_guard``, sensitive tools are gated BEFORE execution.
    With no guard installed (default), behavior is unchanged.
    """
    guard = _tool_guard  # optional process-wide hook; None by default
    if guard is not None:
        verdict = guard.check_tool_call(name, args or {})
        if verdict.blocked:
            return f"ERROR: {verdict.reason}"
    entry = TOOLS.get(name)
    if entry is None:
        return f"ERROR: unknown tool {name!r}. Available: {sorted(TOOLS)}"
    fn, _ = entry
    try:
        return str(fn(**args))
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:  # never let a tool crash the loop
        return f"ERROR: tool {name} failed: {exc}"
