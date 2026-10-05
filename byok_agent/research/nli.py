"""Natural-language-inference checker for the verification gate.

The default is an LLM entailment judge — honest about what it is (a model
call, metered and budgeted like every other). An optional
sentence-transformers cross-encoder exists for operators who install the
heavy dependency themselves; it is NEVER a hard dependency and the README
documents its weight honestly. Evidence is chunked to 512 tokens max per
chunk (the cross-encoder's hard input limit); the chunker applies to both
judges so verdicts are comparable.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


class NLIVerdict:
    """Possible outcomes of checking one claim against one evidence chunk."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"

    ALL = (SUPPORTED, CONTRADICTED, UNVERIFIABLE)


@runtime_checkable
class NLIModel(Protocol):
    """Decide whether ``evidence`` supports, contradicts, or says nothing
    about ``claim``."""

    async def entailment(self, claim: str, evidence: str) -> str:
        """Return one of :class:`NLIVerdict`'s values."""
        ...


def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` tokens.

    Token estimate is ``len // 4`` (codebase convention). Splits prefer
    paragraph boundaries, then sentence boundaries, and never exceed the
    limit — the cross-encoder silently truncates past 512 tokens, which
    would make verdicts depend on invisible input.
    """
    if not text or not text.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    for para in paras:
        units = [para]
        # Split oversized paragraphs on sentence boundaries.
        if len(para) // 4 > max_tokens:
            units = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", para)
                if s.strip()
            ]
        for unit in units:
            toks = max(1, len(unit) // 4)
            if current_tokens + toks > max_tokens and current:
                flush()
            # A single unit larger than the limit is hard-cut (rare; keeps
            # the invariant that no chunk exceeds max_tokens).
            while len(unit) // 4 > max_tokens:
                cut = max_tokens * 4
                chunks.append(unit[:cut])
                unit = unit[cut:]
            current.append(unit)
            current_tokens += max(1, len(unit) // 4)
    flush()
    return chunks


class LLMJudgeNLI:
    """Default NLI: an LLM entailment judge.

    ``stage_caller`` is the pipeline's budgeted model-call helper (any
    object with ``async call(messages, *, max_tokens, temperature)``).
    Unparseable judge output degrades to UNVERIFIABLE (fail-safe: a claim
    we cannot judge is not a claim we can publish under the strict
    policy).
    """

    _PARSE = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNVERIFIABLE)\b", re.I)

    def __init__(self, stage_caller: object) -> None:
        self._caller = stage_caller

    async def entailment(self, claim: str, evidence: str) -> str:
        prompt = (
            "You are a strict fact-checking judge. Read the EVIDENCE, then "
            "the CLAIM. Answer with exactly one word:\n"
            "- SUPPORTED: the evidence directly states or clearly implies the claim.\n"
            "- CONTRADICTED: the evidence directly states the opposite of the claim.\n"
            "- UNVERIFIABLE: the evidence neither supports nor contradicts the claim.\n\n"
            f"EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n\nAnswer:"
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        text = result.text or ""
        # Negation guard: "NOT SUPPORTED" contains "SUPPORTED" as a word —
        # matching the keyword naively would flip the verdict. A negated
        # verdict means the judge couldn't decide => UNVERIFIABLE.
        if re.search(r"\bnot\s+(supported|contradicted|unverifiable)\b", text, re.I):
            log.warning("NLI judge negated its verdict; treating as unverifiable")
            return NLIVerdict.UNVERIFIABLE
        m = self._PARSE.search(text)
        if not m:
            log.warning("NLI judge output unparseable; treating as unverifiable")
            return NLIVerdict.UNVERIFIABLE
        return m.group(1).lower()


class CrossEncoderNLI:
    """Optional NLI via a sentence-transformers cross-encoder.

    Constructing this requires ``sentence-transformers`` (and torch) to be
    installed — it is NOT in requirements.txt and never will be as a hard
    dependency. Raises :class:`ImportError` with install instructions when
    missing. The default model is ``cross-encoder/nli-deberta-v3-base``
    (512-token limit — enforced by :func:`chunk_text` before calling here).
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderNLI needs 'sentence-transformers' installed "
                "(pip install sentence-transformers). The default "
                "LLMJudgeNLI works without it."
            ) from exc
        self._model = CrossEncoder(model_name)
        # nli-deberta-v3-base label order: contradiction, neutral, entailment
        self._labels = (
            NLIVerdict.CONTRADICTED,
            NLIVerdict.UNVERIFIABLE,
            NLIVerdict.SUPPORTED,
        )

    async def entailment(self, claim: str, evidence: str) -> str:
        scores = self._model.predict([(evidence, claim)])
        best = int(scores.argmax())
        return self._labels[best]
