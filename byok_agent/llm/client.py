"""Model-agnostic LLM client layer.

All provider access goes through the :class:`LLMClient` protocol. The
concrete :class:`LiteLLMClient` delegates to the ``litellm`` package, which
normalises the OpenAI / Anthropic / DeepSeek / Ollama APIs behind one call
shape. Model strings follow litellm conventions, e.g.::

    openai/gpt-4o
    anthropic/claude-sonnet-4-20250514
    deepseek/deepseek-chat
    ollama/llama3

The provider segment is everything before the first ``/``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# -- secret sanitization -------------------------------------------------
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI-style keys and friends: sk-... (also sk-ant-, sk-proj-).
    re.compile(r"sk-[A-Za-z0-9\-_]{8,}"),
    # "api_key": "..." / api-key=... shapes.
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)([^\s'\"]{4,})"),
    # Authorization: Bearer <token>
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-_\.]{8,})"),
)


def sanitize_secret_text(text: str) -> str:
    """Redact key-like material from error/log text.

    Provider SDKs love to echo request context (headers, bodies) inside
    exceptions. Never let a raw exception reach a user or a log line
    without passing through here first.
    """
    redacted = _SECRET_PATTERNS[0].sub("sk-***REDACTED***", text)
    redacted = _SECRET_PATTERNS[1].sub(r"\1***REDACTED***", redacted)
    redacted = _SECRET_PATTERNS[2].sub(r"\1***REDACTED***", redacted)
    return redacted


@dataclass(frozen=True)
class CompletionResult:
    """Normalised result of a single model call."""

    text: str
    model: str  # model string actually used
    provider: str  # provider segment of the model string
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # prompt-cached input tokens, when reported
    latency_s: float = 0.0
    raw: Any = None  # provider response object — debugging only, never logged


class LLMError(Exception):
    """Raised when the underlying provider call fails."""


def provider_of(model: str) -> str:
    """Return the provider segment of a litellm-style model string."""
    return model.split("/", 1)[0] if "/" in model else "unknown"


@runtime_checkable
class LLMClient(Protocol):
    """Async model-call contract every provider adapter must satisfy."""

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        """Complete a chat message list with ``model``.

        ``api_key`` is supplied per call — decrypted at the call boundary by
        the caller (see :mod:`src.vault`). It is never stored on the client
        and must never be logged.
        """
        ...


class LiteLLMClient:
    """An :class:`LLMClient` backed by ``litellm.acompletion``.

    ``litellm`` is imported lazily so the rest of the platform (vault,
    budgets, tests) works without the heavy dependency installed.
    """

    def __init__(self, *, timeout: float = 120.0) -> None:
        self._timeout = timeout

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> CompletionResult:
        try:
            import litellm
        except Exception as exc:  # ImportError AND env breakage (e.g. bad proxy vars breaking httpx)
            raise LLMError(
                "litellm is not available; install requirements.txt to make "
                f"real model calls ({type(exc).__name__}: {exc})"
            ) from exc

        started = time.monotonic()
        # A caller-supplied timeout kwarg must not collide with ours.
        timeout = kwargs.pop("timeout", self._timeout)
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=messages,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
        except Exception as exc:  # litellm raises provider-specific errors
            # Sanitize: provider SDKs may echo key material in exceptions.
            raise LLMError(
                sanitize_secret_text(f"model call failed for {model!r}: {exc}")
            ) from exc
        latency = time.monotonic() - started

        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            text = ""

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)

        return CompletionResult(
            text=text,
            model=model,
            provider=provider_of(model),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached,
            latency_s=latency,
            raw=resp,
        )

    async def atranscribe(
        self,
        *,
        model: str,
        audio_file: Any,
        api_key: str,
        timeout: float | None = None,
    ) -> str:
        """Transcribe audio via ``litellm.atranscription``.

        Returns the transcript text. Raises :class:`LLMError` (sanitized)
        on failure.
        """
        try:
            import litellm
        except Exception as exc:
            raise LLMError(
                "litellm is not available; install requirements.txt for "
                f"transcription ({type(exc).__name__}: {exc})"
            ) from exc
        try:
            resp = await litellm.atranscription(
                model=model,
                file=audio_file,
                api_key=api_key,
                timeout=timeout or self._timeout,
            )
        except Exception as exc:
            raise LLMError(
                sanitize_secret_text(f"transcription failed for {model!r}: {exc}")
            ) from exc
        try:
            return str(resp.text or "").strip()
        except AttributeError:
            return str(resp or "").strip()
