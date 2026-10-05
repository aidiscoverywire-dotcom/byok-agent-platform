"""Per-key model routing rules.

A routing rule pins a key to a set of model patterns (fnmatch, e.g.
``deepseek/*``, ``ollama/*``, ``openai/gpt-4o-mini``). When at least one
rule exists for a key, the key may ONLY serve models matching >=1 pattern —
everything else is denied fail-closed.

Use case: the cheap key only serves cheap models; the frontier key is the
only one allowed near ``anthropic/*``. The dashboard manages the rules;
:class:`~byok_agent.llm.router.Router` enforces them.
"""
from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RoutingRuleDenied(Exception):
    """A routing target was denied by a per-key routing rule."""


class RoutingRuleStore:
    """DB-backed routing rules."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def add_rule(self, key_id: str, model_pattern: str) -> int:
        pattern = model_pattern.strip()
        if not pattern:
            raise ValueError("model_pattern must not be empty")
        row = await self._db.fetchrow(
            "INSERT INTO routing_rules (key_id, model_pattern, created_at)"
            " VALUES ($1, $2, $3) RETURNING id",
            key_id,
            pattern,
            _utcnow(),
        )
        assert row is not None
        return int(row["id"])

    async def remove_rule(self, rule_id: int) -> bool:
        row = await self._db.fetchrow(
            "DELETE FROM routing_rules WHERE id = $1 RETURNING id", rule_id
        )
        return row is not None

    async def list_rules(
        self, key_id: str | None = None
    ) -> list[dict]:
        if key_id is not None:
            rows = await self._db.fetch(
                "SELECT id, key_id, model_pattern, created_at FROM routing_rules"
                " WHERE key_id = $1 ORDER BY id",
                key_id,
            )
        else:
            rows = await self._db.fetch(
                "SELECT id, key_id, model_pattern, created_at FROM routing_rules"
                " ORDER BY key_id, id"
            )
        return [dict(r) for r in rows]

    async def allowed(self, key_id: str, model: str) -> bool:
        """True if ``model`` may be served by ``key_id``.

        No rules for the key => allowed (no restriction configured).
        """
        rows = await self._db.fetch(
            "SELECT model_pattern FROM routing_rules WHERE key_id = $1", key_id
        )
        if not rows:
            return True
        return any(
            fnmatch.fnmatchcase(model, r["model_pattern"]) for r in rows
        )
