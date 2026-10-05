"""Model-agnostic LLM layer: protocol, litellm adapter, fail-closed router."""
from .client import (
    CompletionResult,
    LLMClient,
    LLMError,
    LiteLLMClient,
    provider_of,
    sanitize_secret_text,
)
from .router import (
    AllTargetsFailed,
    AttemptRecord,
    FallbackNotConsented,
    ProviderMismatchError,
    RouteTarget,
    Router,
)
from .rules import RoutingRuleDenied, RoutingRuleStore

__all__ = [
    "CompletionResult",
    "LLMClient",
    "LLMError",
    "LiteLLMClient",
    "provider_of",
    "sanitize_secret_text",
    "AllTargetsFailed",
    "AttemptRecord",
    "FallbackNotConsented",
    "ProviderMismatchError",
    "RouteTarget",
    "Router",
    "RoutingRuleDenied",
    "RoutingRuleStore",
]
