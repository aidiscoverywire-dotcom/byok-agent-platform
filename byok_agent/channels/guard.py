"""Deterministic guard: injection scanning, DLP, and sensitive-tool gating.

Three layers, all deterministic (no model in the loop):

1. **Inbound scan** — regex patterns for prompt-injection and credential-
   exfiltration attempts. A hit blocks the message BEFORE any agent/tool
   execution and is written to the audit log.
2. **Outbound DLP** — every reply passes through Phase 0's
   ``sanitize_secret_text`` so key fragments never leave over chat.
3. **Sensitive-tool gate** — tools whose names match
   ``SENSITIVE_TOOL_PATTERNS`` (deploy/delete/exec/shell/...) need an
   approver. Over WhatsApp there is no interactive approver, so the
   worker's default approver DENIES (fail-closed) and audits the attempt.
   The upgrade path is a reply-to-approve flow or running the agent on a
   graph with LangGraph ``interrupt()`` gates (the Phase 1 builder graph
   already uses ``interrupt()`` for promotion) — see README.

The gate is enforced via an optional hook in ``byok_agent.agent.tools``:
:func:`install_tool_guard` registers this guard, and ``run_tool`` consults
it before executing. With no guard installed, tool behavior is unchanged
(all Phase 0/1 tests stay green).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

# -- 1. inbound injection patterns ----------------------------------------
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction-override",
        re.compile(
            r"(ignore|disregard|forget|override)\s+"
            r"(all\s+|any\s+|your\s+|previous\s+|prior\s+|earlier\s+)*"
            r"(instructions|instruction|prompts?|rules|guidelines|system)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt-extraction",
        re.compile(
            r"(reveal|show|print|output|dump|disclose)\s+"
            r"(your\s+|the\s+)?(system\s+prompt|initial\s+prompt|instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "role-reassignment",
        re.compile(
            r"you\s+are\s+now\s+(a\s+)?(different|new|evil|unrestricted|jailbroken)",
            re.IGNORECASE,
        ),
    ),
    (
        "credential-exfiltration",
        re.compile(
            r"(send|give|show|print|reveal|exfiltrate|dump)\s+.*"
            r"(api[_-]?key|secret|password|passwd|private[_-]?key|bearer\s+token"
            r"|access[_-]?token|sk-[a-zA-Z0-9]{8,})",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak-marker",
        re.compile(r"\b(DAN|jailbreak|developer\s+mode)\b", re.IGNORECASE),
    ),
)

# -- 3. sensitive tools ------------------------------------------------------
# Tool names matching any of these need human approval. Phase 0/1 tools
# (get_current_time, read_file, web_search_stub, sandbox exec in a
# container) don't match; anything touching real files / deploys will.
SENSITIVE_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"deploy", re.IGNORECASE),
    re.compile(r"delete|remove|rm\b", re.IGNORECASE),
    re.compile(r"\bexec\b|shell|subprocess|os\.system", re.IGNORECASE),
    re.compile(r"write_file|overwrite", re.IGNORECASE),
    re.compile(r"publish|release", re.IGNORECASE),
)


@dataclass(frozen=True)
class GuardVerdict:
    blocked: bool
    reason: str = ""


# Approver: (tool_name, args) -> True if the call may proceed.
Approver = Callable[[str, dict[str, Any]], bool]


def _deny_all(tool: str, args: dict[str, Any]) -> bool:
    return False


class ToolGuard:
    """Deterministic pre/post filters + sensitive-tool gate."""

    def __init__(self, approver: Approver | None = None) -> None:
        # Over WhatsApp there is no interactive approver: deny by default.
        self._approver: Approver = approver or _deny_all
        self.blocked_events: list[dict[str, str]] = []  # observability

    # -- inbound ---------------------------------------------------------
    def check_inbound(self, text: str) -> GuardVerdict:
        """Scan an inbound chat message. Blocked => never reaches the agent."""
        for name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text or ""):
                self.blocked_events.append({"kind": name, "where": "inbound"})
                log.warning("guard: blocked inbound (%s)", name)
                return GuardVerdict(True, f"blocked: {name}")
        return GuardVerdict(False)

    # -- outbound --------------------------------------------------------
    def redact_outbound(self, text: str) -> str:
        """DLP: scrub secret-like fragments from anything we send."""
        from ..llm import sanitize_secret_text

        return sanitize_secret_text(text or "")

    # -- tool calls ------------------------------------------------------
    def is_sensitive(self, tool_name: str) -> bool:
        return any(p.search(tool_name or "") for p in SENSITIVE_TOOL_PATTERNS)

    def check_tool_call(self, name: str, args: dict[str, Any]) -> GuardVerdict:
        """Gate a tool call. Sensitive tools need the approver."""
        if not self.is_sensitive(name):
            return GuardVerdict(False)
        try:
            ok = bool(self._approver(name, args or {}))
        except Exception as exc:  # approver failure => deny
            log.error("guard: approver raised; denying: %s", exc)
            ok = False
        if not ok:
            self.blocked_events.append({"kind": "sensitive-tool", "where": name})
            log.warning("guard: denied sensitive tool %s", name)
            return GuardVerdict(
                True,
                f"sensitive tool '{name}' requires approval; "
                "not available over this channel",
            )
        return GuardVerdict(False)


def install_tool_guard(guard: ToolGuard) -> None:
    """Register a process-wide guard consulted by ``run_tool``."""
    from ..agent import tools as toolmod

    toolmod._tool_guard = guard  # noqa: SLF001 — intentional hook point


def uninstall_tool_guard() -> None:
    """Remove the process-wide guard (tests)."""
    from ..agent import tools as toolmod

    toolmod._tool_guard = None  # noqa: SLF001
