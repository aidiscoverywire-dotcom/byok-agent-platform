"""The verification gate — audited semantics, not the naive version.

Three properties distinguish this gate from a naive "LLM checks itself"
loop:

1. **Independent evidence.** :meth:`VerificationGate.verify_section`
   receives ONLY the hold-out ``verify_evidence`` set. The draft stage's
   evidence is never passed in — the split is made by the pipeline and the
   gate's signature makes reuse structurally impossible. A verifier that
   re-reads the draft's evidence would be circular (the bug this gate
   was built to avoid).

2. **Unverifiable blocks by default.** Under the ``strict`` policy, both
   ``contradicted`` AND ``unverifiable`` claims block publish — most
   hallucinations surface as "unverifiable", not "contradicted", so a gate
   that only blocks contradictions is decorative. ``lenient`` allows
   unverifiable claims through but flags them inline as ``[unverified]``;
   contradicted claims never pass in either mode.

3. **Capped revise loop.** ``revise -> verify`` runs at most
   ``max_rounds`` (default 3). After the cap the article FAILS with
   :class:`VerificationFailed` carrying the full report — never an
   infinite loop, never silent publication of unchecked text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from byok_agent.research.nli import NLIModel, NLIVerdict, chunk_text

log = logging.getLogger(__name__)

MAX_REVISE_ROUNDS = 3


@dataclass
class AtomicClaim:
    """One factual assertion extracted from a section."""

    text: str


@dataclass
class EvidenceChunk:
    """One chunk of hold-out evidence, traceable to its source."""

    text: str
    source_url: str
    source_title: str = ""


@dataclass
class ClaimCheck:
    claim: str
    verdict: str  # one of NLIVerdict.ALL
    evidence_refs: list[str] = field(default_factory=list)  # source URLs consulted


@dataclass
class SectionVerification:
    section_title: str
    passed: bool
    checks: list[ClaimCheck] = field(default_factory=list)
    rounds: int = 1
    blocked_claims: list[str] = field(default_factory=list)


class VerificationFailed(Exception):
    """The revise -> verify loop exhausted its cap with claims still
    blocked. Carries the :class:`SectionVerification` report — the caller
    must surface it, never publish past it."""

    def __init__(self, section_title: str, report: SectionVerification) -> None:
        super().__init__(
            f"verification failed for section {section_title!r} after "
            f"{report.rounds} rounds; blocked claims: {report.blocked_claims}"
        )
        self.section_title = section_title
        self.report = report


class VerificationGate:
    """Claim-level verification against INDEPENDENT evidence.

    Args:
        nli: the entailment judge (LLM judge by default).
        stage_caller: budgeted model-call helper for decomposition/revision.
        policy: ``"strict"`` (default) or ``"lenient"``.
        max_rounds: hard cap on revise -> verify iterations.
    """

    def __init__(
        self,
        *,
        nli: NLIModel,
        stage_caller: Any,
        policy: str = "strict",
        max_rounds: int = MAX_REVISE_ROUNDS,
    ) -> None:
        if policy not in ("strict", "lenient"):
            raise ValueError("policy must be 'strict' or 'lenient'")
        self._nli = nli
        self._caller = stage_caller
        self._policy = policy
        self._max_rounds = max(1, max_rounds)

    # -- claim decomposition ------------------------------------------------
    async def decompose_claims(self, section_text: str) -> list[AtomicClaim]:
        """Split a section into atomic factual claims via the LLM.

        Non-factual content (opinions, transitions, rhetorical questions)
        must NOT become claims — the prompt says so explicitly, and the
        parser tolerates numbered, bulleted, or plain-line output.
        """
        prompt = (
            "[ROLE: DECOMPOSE]\n"
            "Split the SECTION below into atomic FACTUAL claims — one "
            "verifiable assertion per line. Rules:\n"
            "- Skip opinions, transitions, rhetorical questions, and generic\n"
            "  advice with no factual content.\n"
            "- Each claim must be checkable against evidence on its own.\n"
            "- Output one claim per line, no numbering, no bullets.\n\n"
            f"SECTION:\n{section_text}\n\nClaims:"
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        claims: list[AtomicClaim] = []
        for line in (result.text or "").splitlines():
            line = re.sub(r"^[\s\-\*\d\.\)\]]+", "", line).strip()
            if len(line) >= 12:  # ignore fragments and empty lines
                claims.append(AtomicClaim(text=line))
        return claims

    # -- per-claim checking ---------------------------------------------------
    async def check_claim(
        self, claim: AtomicClaim, evidence: list[EvidenceChunk]
    ) -> ClaimCheck:
        """Check one claim against the hold-out evidence chunks.

        Aggregation: any CONTRADICTED chunk wins (a single contradiction
        kills the claim); otherwise any SUPPORTED chunk passes it; with
        neither, the claim is UNVERIFIABLE.
        """
        refs: list[str] = []
        saw_supported = False
        for chunk in evidence:
            verdict = await self._nli.entailment(claim.text, chunk.text)
            if chunk.source_url not in refs:
                refs.append(chunk.source_url)
            if verdict == NLIVerdict.CONTRADICTED:
                return ClaimCheck(
                    claim=claim.text, verdict=verdict, evidence_refs=refs
                )
            if verdict == NLIVerdict.SUPPORTED:
                saw_supported = True
        return ClaimCheck(
            claim=claim.text,
            verdict=NLIVerdict.SUPPORTED if saw_supported else NLIVerdict.UNVERIFIABLE,
            evidence_refs=refs,
        )

    # -- section gate ---------------------------------------------------------
    def _blocked(self, verdict: str) -> bool:
        if verdict == NLIVerdict.CONTRADICTED:
            return True  # never publishable, either policy
        if verdict == NLIVerdict.UNVERIFIABLE:
            return self._policy == "strict"
        return False

    async def verify_section(
        self,
        title: str,
        section_text: str,
        verify_evidence: list[EvidenceChunk],
        revise_fn: Callable[[str, str, str], Awaitable[str]],
    ) -> tuple[str, SectionVerification]:
        """Verify one section, revising up to ``max_rounds`` times.

        ``verify_evidence`` MUST be the hold-out set — disjoint from the
        draft stage's evidence. The gate never sees the draft evidence;
        pass the wrong set and the gate is circular (see module docstring).

        Returns ``(final_text, report)``. Under ``lenient`` policy,
        surviving unverifiable claims are flagged inline as ``[unverified]``.
        Raises :class:`VerificationFailed` when the cap is exhausted with
        claims still blocked.
        """
        text = section_text
        last_report = SectionVerification(section_title=title, passed=False)
        for round_no in range(1, self._max_rounds + 1):
            claims = await self.decompose_claims(text)
            checks = [await self.check_claim(c, verify_evidence) for c in claims]
            blocked = [ch.claim for ch in checks if self._blocked(ch.verdict)]
            last_report = SectionVerification(
                section_title=title,
                passed=not blocked,
                checks=checks,
                rounds=round_no,
                blocked_claims=blocked,
            )
            if not blocked:
                final = text
                if self._policy == "lenient":
                    final = self._flag_unverified(text, checks)
                return final, last_report
            if round_no >= self._max_rounds:
                break
            feedback = self._revision_feedback(checks)
            log.info(
                "section %r failed verification (round %d/%d); revising",
                title, round_no, self._max_rounds,
            )
            text = await revise_fn(title, text, feedback)
        raise VerificationFailed(title, last_report)

    @staticmethod
    def _revision_feedback(checks: list[ClaimCheck]) -> str:
        lines = []
        for ch in checks:
            if ch.verdict == NLIVerdict.CONTRADICTED:
                lines.append(f"- CONTRADICTED by evidence: {ch.claim}")
            elif ch.verdict == NLIVerdict.UNVERIFIABLE:
                lines.append(f"- UNVERIFIABLE (no evidence found): {ch.claim}")
        return "\n".join(lines)

    @staticmethod
    def _flag_unverified(text: str, checks: list[ClaimCheck]) -> str:
        """Append [unverified] markers for surviving unverifiable claims
        (lenient policy only). Marks the first sentence containing a
        distinctive fragment of each unverifiable claim."""
        flagged = text
        for ch in checks:
            if ch.verdict != NLIVerdict.UNVERIFIABLE:
                continue
            fragment = ch.claim[:40].strip()
            if not fragment:
                continue
            marker = fragment + " [unverified]"
            # Per-claim idempotence: skip only if THIS claim was already
            # flagged (the old `"[unverified]" not in flagged` check stopped
            # after the first claim — every later claim went unmarked).
            if marker not in flagged and fragment in flagged:
                flagged = flagged.replace(fragment, marker, 1)
        return flagged


def build_evidence_chunks(
    results: list[Any], *, max_tokens: int = 512
) -> list[EvidenceChunk]:
    """Chunk search results into evidence chunks (traceable to source URL).

    Accepts :class:`~byok_agent.research.search.SearchResult` objects
    (or anything with ``content``, ``snippet``, ``url``, ``title``).
    Snippet-only hits (e.g. Brave) still produce one chunk — weaker
    evidence, but evidence.
    """
    chunks: list[EvidenceChunk] = []
    for r in results:
        text = (r.content or r.snippet or "").strip()
        if not text:
            continue
        for piece in chunk_text(text, max_tokens=max_tokens):
            chunks.append(
                EvidenceChunk(
                    text=piece,
                    source_url=r.url,
                    source_title=r.title,
                )
            )
    return chunks


def split_evidence(
    results: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Split search results into two DISJOINT sets: draft evidence and
    hold-out verify evidence.

    Interleaved by rank (even indices -> draft, odd -> verify) so both
    sets cover the topic's breadth instead of one set getting only the
    top hits. Disjoint by construction: no result object appears in both
    lists. The verification gate receives ONLY the verify set.
    """
    draft = [r for i, r in enumerate(results) if i % 2 == 0]
    verify = [r for i, r in enumerate(results) if i % 2 == 1]
    assert not (set(map(id, draft)) & set(map(id, verify))), "evidence sets overlap!"
    return draft, verify
