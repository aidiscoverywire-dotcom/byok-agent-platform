"""Fail-closed model routing with consent-gated fallback.

The router tries the primary ``(key_id, model)`` first and, only on
:class:`~byok_agent.llm.client.LLMError`, walks the fallback list in order.

Binding rule: a key may ONLY serve models whose provider matches the key
record's ``provider``. A mismatch is a configuration error, never silently
bridged — the client's Anthropic key is never sent to OpenAI.

Consent rule (fail closed): a fallback whose provider differs from the
primary's provider is a *cross-provider* fallback. It is attempted only if
the fallback key's vault record has ``allow_cross_provider_fallback=True``.
``provider_of()`` returning ``"unknown"`` is treated as cross-provider
(requiring consent), never as same-provider — fail closed, not fail open.

Failure aggregation: a failed primary never hides behind an immediate
``FallbackNotConsented``. Non-consented / mismatched fallbacks are skipped
(and recorded); remaining consented fallbacks are still tried. Only when
every target is exhausted is :class:`AllTargetsFailed` raised, carrying the
primary's error message (sanitized) and every other outcome.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .client import (
    CompletionResult,
    LLMClient,
    LLMError,
    provider_of,
    sanitize_secret_text,
)
from .rules import RoutingRuleDenied, RoutingRuleStore

log = logging.getLogger(__name__)


class FallbackNotConsented(Exception):
    """Legacy: kept for API compatibility.

    The router no longer raises this mid-loop (see :class:`AllTargetsFailed`
    — skipped fallbacks are aggregated instead). Retained so existing
    ``except FallbackNotConsented`` handlers keep working if anyone raises
    it in the future.
    """


class ProviderMismatchError(LLMError):
    """A key was asked to serve a model from a different provider.

    Raised fail-fast for the PRIMARY target: e.g. an ``anthropic`` key with
    model ``openai/gpt-4o-mini``. Fallback mismatches are skipped, not
    raised, so remaining fallbacks still get their chance.
    """


class AllTargetsFailed(LLMError):
    """Every routing target failed or was skipped.

    Attributes:
        outcomes: list of ``(RouteTarget, outcome)`` where outcome is the
            sanitized error message, or ``"skipped: <reason>"`` for targets
            that were never attempted.
    """

    def __init__(self, outcomes: list[tuple["RouteTarget", str]]) -> None:
        self.outcomes = list(outcomes)
        detail = "; ".join(f"{t.model} [{t.key_id}]: {why}" for t, why in outcomes)
        super().__init__(f"all {len(outcomes)} routing target(s) failed: {detail}")


@dataclass(frozen=True)
class RouteTarget:
    """One routable endpoint: whose key, which model."""

    key_id: str
    model: str


@dataclass
class AttemptRecord:
    """One failed or skipped routing attempt, for ledger metering."""

    key_id: str
    provider: str  # provider_of(target.model); "unknown" if unparseable
    model: str
    status: str  # "failed" | "skipped"
    error: str  # sanitized reason


AttemptSink = Callable[[AttemptRecord], Awaitable[None]]


def _same_provider(a: str, b: str) -> bool:
    """Fail-closed provider equality.

    ``"unknown"`` (unparseable model string) is NEVER equal to anything —
    not even another ``"unknown"``. Unknown means "can't prove same
    vendor", which under BYOK means consent is required.
    """
    if a == "unknown" or b == "unknown":
        return False
    return a == b


def _check_binding(record: Any, model: str) -> None:
    """Raise :class:`ProviderMismatchError` if the key cannot serve the model."""
    model_provider = provider_of(model)
    if model_provider == "unknown" or record.provider != model_provider:
        raise ProviderMismatchError(
            f"key {record.key_id!r} belongs to provider {record.provider!r} "
            f"but model {model!r} needs provider {model_provider!r}; "
            "use a 'provider/model' string matching the key's provider"
        )


class Router:
    """Routes model calls across keys with consent-gated fallback.

    The router is deliberately thin: key plaintext is fetched from the vault
    at the call boundary, used once, and never retained. Callers meter every
    attempt (including failures) via ``attempt_sink``.
    """

    def __init__(self, client: LLMClient, vault: Any) -> None:
        """
        Args:
            client: An :class:`LLMClient` (normally :class:`LiteLLMClient`).
            vault: A :class:`~byok_agent.vault.store.SecretStore` (typed as
                Any to avoid a hard import cycle; must expose ``get_key``
                and ``get_record``, both accepting an ``owner`` kwarg).
        """
        self._client = client
        self._vault = vault
        self._rule_store: RoutingRuleStore | None = None

    def set_rule_store(self, store: RoutingRuleStore | None) -> None:
        """Attach a routing-rule store (dashboard-managed). None disables."""
        self._rule_store = store

    async def _check_routing_rules(self, key_id: str, model: str) -> None:
        """Raise :class:`RoutingRuleDenied` if rules forbid this pairing."""
        store = self._rule_store
        if store is not None and not await store.allowed(key_id, model):
            raise RoutingRuleDenied(
                f"routing rule denied: key {key_id!r} may not serve model"
                f" {model!r} (see routing_rules)"
            )

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        primary: RouteTarget,
        fallbacks: tuple[RouteTarget, ...] = (),
        *,
        owner: str | None = None,
        attempt_sink: AttemptSink | None = None,
        **kwargs: Any,
    ) -> tuple[CompletionResult, RouteTarget]:
        """Complete via primary, falling back only with consent.

        Args:
            owner: tenant scope — the key must belong to this owner or the
                target is skipped as if unknown (no existence oracle).
            attempt_sink: optional async callback receiving an
                :class:`AttemptRecord` for every FAILED or SKIPPED attempt
                (the successful call is NOT sunk — the caller meters it).

        Returns the result together with the target that actually served it
        (so the caller meters the correct ``key_id``).

        Raises:
            ProviderMismatchError: primary key/model provider mismatch
                (fail fast — configuration error).
            AllTargetsFailed: every target failed or was skipped; carries
                all sanitized error messages including the primary's.
        """
        # Fail fast on primary misconfiguration — before spending anything.
        primary_record = await self._vault.get_record(primary.key_id, owner=owner)
        _check_binding(primary_record, primary.model)
        # Routing rules are configuration too: a denied primary fails fast.
        await self._check_routing_rules(primary.key_id, primary.model)
        primary_provider = provider_of(primary.model)

        outcomes: list[tuple[RouteTarget, str]] = []

        for i, target in enumerate((primary, *fallbacks)):
            is_fallback = i > 0
            if is_fallback:
                try:
                    record = await self._vault.get_record(target.key_id, owner=owner)
                except Exception as exc:  # KeyNotFoundError / RevokedKeyError
                    why = f"vault: {sanitize_secret_text(str(exc))}"
                    outcomes.append((target, f"skipped: {why}"))
                    await self._sink(attempt_sink, target, "skipped", why)
                    continue
                target_provider = provider_of(target.model)
                if target_provider == "unknown" or record.provider != target_provider:
                    why = (
                        f"provider mismatch: key is {record.provider!r}, "
                        f"model needs {target_provider!r}"
                    )
                    outcomes.append((target, f"skipped: {why}"))
                    await self._sink(attempt_sink, target, "skipped", why)
                    continue
                if self._rule_store is not None and not await self._rule_store.allowed(
                    target.key_id, target.model
                ):
                    why = (
                        f"routing rule denied: key {target.key_id!r} may not"
                        f" serve model {target.model!r}"
                    )
                    outcomes.append((target, f"skipped: {why}"))
                    await self._sink(attempt_sink, target, "skipped", why)
                    continue
                if not _same_provider(target_provider, primary_provider):
                    if not record.allow_cross_provider_fallback:
                        why = (
                            f"cross-provider fallback {primary_provider!r} -> "
                            f"{target_provider!r} blocked: key {target.key_id!r} "
                            "has allow_cross_provider_fallback=False"
                        )
                        outcomes.append((target, f"skipped: {why}"))
                        await self._sink(attempt_sink, target, "skipped", why)
                        continue
                    log.warning(
                        "cross-provider fallback consented: %s -> %s (key %s)",
                        primary_provider,
                        target_provider,
                        target.key_id,
                    )
                else:
                    log.warning(
                        "same-provider fallback: key %s model %s",
                        target.key_id,
                        target.model,
                    )

            # PRIMARY vault failures (unknown/revoked key) propagate fail-fast:
            # a dead primary is a hard error, not something to route around
            # silently. Fallback vault failures skip to the next target.
            try:
                api_key = await self._vault.get_key(target.key_id, owner=owner)
            except Exception as exc:  # KeyNotFoundError / RevokedKeyError
                if not is_fallback:
                    raise
                why = f"vault: {sanitize_secret_text(str(exc))}"
                outcomes.append((target, f"skipped: {why}"))
                await self._sink(attempt_sink, target, "skipped", why)
                continue
            try:
                result = await self._client.acomplete(
                    messages, target.model, api_key=api_key, **kwargs
                )
            except LLMError as exc:
                why = sanitize_secret_text(str(exc))
                outcomes.append((target, why))
                await self._sink(attempt_sink, target, "failed", why)
                log.warning("target %s failed: %s", target, why)
                continue
            finally:
                # Do not retain the plaintext: drop our reference immediately.
                # (CPython refcounting frees it; no stronger guarantee is
                #  possible in pure Python — see README security notes.)
                del api_key
            return result, target

        raise AllTargetsFailed(outcomes)

    @staticmethod
    async def _sink(
        sink: AttemptSink | None, target: RouteTarget, status: str, error: str
    ) -> None:
        if sink is None:
            return
        await sink(
            AttemptRecord(
                key_id=target.key_id,
                provider=provider_of(target.model),
                model=target.model,
                status=status,
                error=error,
            )
        )
