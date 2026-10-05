"""Research pipeline: plan -> gather -> outline -> draft -> verify -> SEO.

STORM-inspired stages on a GPT-Researcher-style skeleton. Every model
call goes through the Phase 0 budget gates (``before_model_call`` /
``record_model_call`` / ``record_stream_abort`` — the exact pattern from
``byok_agent.agent.graph``); every search call is metered in the Phase 0
ledger with an ESTIMATE cost.

Evidence discipline (the audited property): ``gather`` splits results into
two DISJOINT sets via :func:`split_evidence`. The draft stage sees ONLY
the draft set; :class:`VerificationGate` receives ONLY the hold-out
verify set. Disjoint by construction — see the assertion in
``split_evidence``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from byok_agent.ledger import estimate_cost, estimate_max_cost_usd
from byok_agent.research.nli import LLMJudgeNLI
from byok_agent.research.search import (
    BraveProvider,
    ExaProvider,
    SEARCH_PROVIDER_NAMES,
    SearchProvider,
    SearchResult,
    TavilyProvider,
    dedupe_results,
    find_search_key,
)
from byok_agent.research.seo import (
    assemble_article,
    faq_markdown,
    keyword_density,
    keyword_in_first_n_words,
    meta_description,
    seo_title,
)
from byok_agent.research.verify import (
    EvidenceChunk,
    SectionVerification,
    VerificationFailed,
    VerificationGate,
    build_evidence_chunks,
    split_evidence,
)

log = logging.getLogger(__name__)

DEPTH_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {"queries": 3, "results_per_query": 5, "sections": 4},
    "standard": {"queries": 4, "results_per_query": 8, "sections": 6},
    "deep": {"queries": 6, "results_per_query": 10, "sections": 8},
}

PROMPTS_DIR = Path(__file__).parent / "prompts"


def render_template(name: str, *, prompts_dir: Path | None = None, **kwargs: Any) -> str:
    """Render a ``{{var}}`` prompt template. Missing files raise; missing
    variables are left as-is (never silently dropped)."""
    directory = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    text = (directory / name).read_text(encoding="utf-8")

    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(kwargs.get(key, match.group(0)))

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, text)


# -- pipeline data ------------------------------------------------------------

@dataclass
class ResearchBrief:
    topic: str
    primary_keyword: str
    queries: list[str] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)
    outline: list[dict[str, str]] = field(default_factory=list)  # [{title, brief}]
    brief_md: str = ""
    gathered_at: str = ""


@dataclass
class ArticleResult:
    topic: str
    primary_keyword: str
    article_md: str
    sources_md: str
    brief_md: str
    verification_reports: list[SectionVerification] = field(default_factory=list)
    seo: dict[str, Any] = field(default_factory=dict)
    spend_usd_est: float = 0.0


# -- budgeted model-call helper -------------------------------------------------

class _StageCaller:
    """One budgeted model call, following the exact Phase 0 pattern.

    Gate BEFORE (with cost reservation), meter EVERY attempt (failed ones
    included, via attempt_sink), release the reservation on failure,
    commit actuals + ledger row on success.
    """

    def __init__(
        self,
        *,
        router: Any,
        primary: Any,
        owner: str,
        enforcer: Any,
        ledger: Any,
        model: str,
    ) -> None:
        self._router = router
        self._primary = primary
        self._owner = owner
        self._enforcer = enforcer
        self._ledger = ledger
        self._model = model

    async def call(
        self, messages: list[dict[str, str]], *, max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Any:
        async def attempt_sink(attempt: Any) -> None:
            await self._ledger.record(
                key_id=attempt.key_id,
                provider=attempt.provider,
                model=attempt.model,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                status=attempt.status,
                error=attempt.error,
            )

        est_in = sum(len(m.get("content", "")) // 4 for m in messages)
        est_cost = estimate_max_cost_usd(self._model, est_in, max_tokens)
        await self._enforcer.before_model_call(
            est_input_tokens=est_in,
            est_max_output_tokens=max_tokens,
            est_max_cost_usd=est_cost,
        )
        try:
            result, target = await self._router.acomplete(
                messages,
                self._primary,
                owner=self._owner,
                attempt_sink=attempt_sink,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            await self._enforcer.record_stream_abort(
                reserved_usd=est_cost,
                reason="research stage model call failed",
            )
            raise
        cost = estimate_cost(target.model, result.raw)
        await self._enforcer.record_model_call(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
            reserved_usd=est_cost,
        )
        await self._ledger.record(
            key_id=target.key_id,
            provider=result.provider,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            estimated_cost_usd=cost,
        )
        return result


# -- the pipeline ---------------------------------------------------------------

class ResearchPipeline:
    """End-to-end research + article pipeline.

    Args:
        router: Phase 0 :class:`Router` (vault-backed, tenant-scoped).
        primary: :class:`RouteTarget` for the LLM key.
        owner: tenant scope for keys.
        vault / ledger / enforcer: Phase 0 components.
        model: litellm-style model string for all stage calls.
        search_providers: provider instances in priority order (resolved
            per run against the vault — first configured wins).
        prompts_dir: override for the default prompt templates.
        nli: entailment judge (defaults to the LLM judge).
    """

    def __init__(
        self,
        *,
        router: Any,
        primary: Any,
        owner: str,
        vault: Any,
        ledger: Any,
        enforcer: Any,
        model: str,
        search_providers: list[SearchProvider] | None = None,
        prompts_dir: Path | None = None,
        nli: Any = None,
    ) -> None:
        self._router = router
        self._primary = primary
        self._owner = owner
        self._vault = vault
        self._ledger = ledger
        self._enforcer = enforcer
        self._model = model
        self._providers = (
            search_providers
            if search_providers is not None
            else [TavilyProvider(), ExaProvider(), BraveProvider()]
        )
        self._prompts_dir = prompts_dir
        self._caller = _StageCaller(
            router=router, primary=primary, owner=owner,
            enforcer=enforcer, ledger=ledger, model=model,
        )
        self._nli = nli or LLMJudgeNLI(self._caller)

    # -- provider resolution ----------------------------------------------------
    async def _resolve_provider(self) -> tuple[SearchProvider, Any]:
        """First configured provider in priority order. Raises a clean
        error naming the exact CLI command when none is configured."""
        for provider in self._providers:
            if provider.name not in SEARCH_PROVIDER_NAMES:
                continue
            record = await find_search_key(self._vault, self._owner, provider.name)
            if record is not None:
                return provider, record
        raise SearchNotConfigured(
            "no search provider configured for this user — run: "
            "byok add-key --provider tavily --label <label> "
            "(or exa / brave)"
        )

    async def _metered_search(
        self,
        provider: SearchProvider,
        record: Any,
        query: str,
        max_results: int,
        run_cap_usd: float,
    ) -> list[SearchResult]:
        """One search call: reserve -> call -> meter -> release. The
        plaintext key is decrypted at the call boundary and never logged.

        ``run_cap_usd`` is the LIFETIME-scale cap ``reserve_key`` expects:
        the caller passes ``baseline_spend + run_budget`` so the check
        measures THIS RUN's spend against the run budget — not the key's
        lifetime spend (which would instantly fail for any reused key whose
        history already exceeds the run budget).
        """
        from byok_agent.vault import VaultError  # local: avoid import cycle at module load

        est = provider.estimate_cost_usd(max_results)
        await self._ledger.reserve_key(record.key_id, est, run_cap_usd)
        try:
            api_key = await self._vault.get_key(record.key_id, owner=self._owner)
            try:
                results = await provider.search(
                    query, max_results, api_key=api_key
                )
            finally:
                del api_key
        except Exception as exc:
            await self._ledger.release_key_reservation(record.key_id, est)
            raise
        await self._ledger.record(
            key_id=record.key_id,
            provider=provider.name,
            model="search-api",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=est,  # ESTIMATE, not billing truth
        )
        await self._ledger.release_key_reservation(record.key_id, est)
        return results

    # -- stage 1: plan ------------------------------------------------------------
    async def plan(self, topic: str, *, n_queries: int) -> list[str]:
        prompt = (
            "[ROLE: PLAN]\n"
            f"Generate {n_queries} distinct web-search queries that together "
            f"cover the topic from different angles (definition, how it works, "
            f"recent developments, expert opinion, comparisons, practical use). "
            f"Output one query per line, no numbering.\n\nTopic: {topic}"
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}], max_tokens=512, temperature=0.5
        )
        queries = [
            re.sub(r"^[\s\-\*\d\.\)\]]+", "", line).strip()
            for line in (result.text or "").splitlines()
        ]
        queries = [q for q in queries if len(q) >= 4][:n_queries]
        if not queries:
            queries = [topic]
        return queries

    # -- stage 2: gather ------------------------------------------------------------
    async def gather(
        self,
        queries: list[str],
        *,
        results_per_query: int,
        max_usd: float,
    ) -> list[SearchResult]:
        provider, record = await self._resolve_provider()
        log.info("searching via %s (%d queries)", provider.name, len(queries))
        # Run-scoped search budget: reserve_key() compares against the key's
        # LIFETIME committed spend, so the cap must be baseline + run budget.
        # Without this, any key whose history already exceeds max_usd would
        # fail every search immediately.
        baseline = await self._ledger.get_spend(record.key_id)
        run_cap = baseline + max_usd
        per_query = await asyncio.gather(
            *[
                self._metered_search(provider, record, q, results_per_query, run_cap)
                for q in queries
            ]
        )
        merged: list[SearchResult] = []
        for hits in per_query:
            merged.extend(hits)
        return dedupe_results(merged)

    # -- stage 3: outline ------------------------------------------------------------
    async def outline(
        self, topic: str, sources: list[SearchResult], *, n_sections: int
    ) -> list[dict[str, str]]:
        src_lines = "\n".join(
            f"- {s.title} ({s.url}): {(s.content or s.snippet)[:200]}"
            for s in sources[:20]
        )
        prompt = (
            "[ROLE: OUTLINE]\n"
            f"Draft an outline of {n_sections} sections for an article on "
            f"'{topic}', grounded ONLY in the sources below. Reply as JSON: "
            f'{{"sections": [{{"title": "...", "brief": "..."}}]}}.\n\n'
            f"SOURCES:\n{src_lines}"
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}], max_tokens=1024, temperature=0.4
        )
        try:
            data = json.loads(result.text)
            sections = [
                {"title": s["title"], "brief": s.get("brief", "")}
                for s in data.get("sections", [])
            ]
            if sections:
                return sections[:n_sections]
        except Exception:
            log.warning("outline JSON unparseable; falling back to default sections")
        return [
            {"title": "Overview", "brief": f"what {topic} is"},
            {"title": "Key details", "brief": "the important facts"},
            {"title": "What to do", "brief": "practical takeaways"},
            {"title": "Caveats", "brief": "limits and uncertainties"},
        ][:n_sections]

    # -- research brief (plan+gather+outline, rendered through the template) -----------
    async def research_brief(
        self, topic: str, *, depth: str = "standard", max_usd: float = 5.0,
        primary_keyword: str = "",
    ) -> ResearchBrief:
        preset = DEPTH_PRESETS[depth]
        queries = await self.plan(topic, n_queries=preset["queries"])
        sources = await self.gather(
            queries, results_per_query=preset["results_per_query"], max_usd=max_usd
        )
        outline = await self.outline(topic, sources, n_sections=preset["sections"])
        brief_md = render_template(
            "research_brief.md",
            prompts_dir=self._prompts_dir,
            topic=topic,
            primary_keyword=primary_keyword or topic,
            date=datetime.now(timezone.utc).date().isoformat(),
        )
        # Append the machine-gathered material as an appendix the writer can use.
        appendix = ["", "## MACHINE-GATHERED APPENDIX (pipeline output)", ""]
        appendix.append("### Search queries used")
        appendix += [f"- {q}" for q in queries]
        appendix.append("")
        appendix.append("### Sources gathered")
        for i, s in enumerate(sources, 1):
            appendix.append(f"{i}. {s.title} — {s.url} [{s.provider}]")
        appendix.append("")
        appendix.append("### Draft outline")
        for s in outline:
            appendix.append(f"- **{s['title']}**: {s['brief']}")
        return ResearchBrief(
            topic=topic,
            primary_keyword=primary_keyword or topic,
            queries=queries,
            sources=sources,
            outline=outline,
            brief_md=brief_md + "\n" + "\n".join(appendix) + "\n",
            gathered_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- stage 4: draft (parallel per-section) ----------------------------------------
    async def _draft_section(
        self,
        topic: str,
        section: dict[str, str],
        draft_evidence: list[EvidenceChunk],
        source_index: dict[str, int],
    ) -> str:
        ev_lines = []
        for ch in draft_evidence[:12]:
            n = source_index.get(ch.source_url, 0)
            ev_lines.append(f"[{n}] {ch.source_title} ({ch.source_url}): {ch.text[:600]}")
        prompt = (
            "[ROLE: DRAFT]\n"
            f"Write the '{section['title']}' section of an article on '{topic}'. "
            f"Section brief: {section['brief']}. Use ONLY the evidence below; "
            f"cite facts inline as [n] matching the source numbers. If the "
            f"evidence does not support a point, say so instead of inventing it.\n\n"
            + "\n\n".join(ev_lines)
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.4
        )
        return result.text or ""

    # -- stage 5: verify is owned by VerificationGate; revision callback ---------------
    async def _revise(
        self, title: str, section_text: str, feedback: str
    ) -> str:
        prompt = (
            "[ROLE: REVISE]\n"
            f"Revise the '{title}' section below to fix every flagged claim. "
            f"Remove claims contradicted by evidence; soften or remove claims "
            f"with no evidence. Keep citations [n] accurate.\n\n"
            f"FEEDBACK:\n{feedback}\n\nSECTION:\n{section_text}\n\nRevised section:"
        )
        result = await self._caller.call(
            [{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.3
        )
        return result.text or section_text

    # -- full article run ----------------------------------------------------------------
    async def run_article(
        self,
        topic: str,
        *,
        depth: str = "standard",
        policy: str = "strict",
        primary_keyword: str = "",
        max_usd: float = 10.0,
    ) -> ArticleResult:
        brief = await self.research_brief(
            topic, depth=depth, max_usd=max_usd, primary_keyword=primary_keyword
        )
        keyword = brief.primary_keyword

        # Disjoint evidence split: draft sees ONLY draft_results,
        # the gate sees ONLY verify_results. Never mixed.
        draft_results, verify_results = split_evidence(brief.sources)
        draft_evidence = build_evidence_chunks(draft_results)
        verify_evidence = build_evidence_chunks(verify_results)

        # Global source numbering for [n] citations.
        source_index = {s.url: i + 1 for i, s in enumerate(brief.sources)}

        # Draft sections in parallel (each gated by the shared budget).
        draft_texts = await asyncio.gather(
            *[
                self._draft_section(topic, sec, draft_evidence, source_index)
                for sec in brief.outline
            ]
        )

        # Verify each section against the HOLD-OUT set only.
        gate = VerificationGate(
            nli=self._nli, stage_caller=self._caller, policy=policy
        )
        final_sections: list[tuple[str, str]] = []
        reports: list[SectionVerification] = []
        for sec, text in zip(brief.outline, draft_texts):
            final_text, report = await gate.verify_section(
                sec["title"], text, verify_evidence, self._revise
            )
            final_sections.append((sec["title"], final_text))
            reports.append(report)

        # SEO pass (pure functions) + FAQ from question queries.
        faq_pairs = self._faq_from_brief(brief)
        title = seo_title(keyword, topic)
        # All outline sections become H2 sections; the lede paragraph is
        # the first section's body, so the keyword-in-first-100-words
        # contract is checked against the full body.
        body = assemble_article(
            title=title,
            intro="",
            sections=list(final_sections),
            faq_md=faq_markdown(faq_pairs) if faq_pairs else "",
            sources_md=self._sources_md(brief.sources),
        )
        # Humanize through the article template? No — the template IS the
        # writer's instruction set; the pipeline already drafted with its
        # rules baked into the draft prompt. The template ships for
        # operator-driven runs and CLI --template overrides.

        meta = meta_description(body[:500], keyword)
        density = keyword_density(body, keyword)
        seo_info = {
            "title": title,
            "title_len": len(title),
            "meta_description": meta,
            "meta_len": len(meta),
            "keyword_density_pct": round(density, 2),
            "keyword_in_first_100": keyword_in_first_n_words(body, keyword),
            "policy": policy,
            "depth": depth,
        }
        spend = await self._spend_so_far()
        return ArticleResult(
            topic=topic,
            primary_keyword=keyword,
            article_md=body,
            sources_md=self._sources_md(brief.sources),
            brief_md=brief.brief_md,
            verification_reports=reports,
            seo=seo_info,
            spend_usd_est=spend,
        )

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _sources_md(sources: list[SearchResult]) -> str:
        lines = []
        for i, s in enumerate(sources, 1):
            date = f" ({s.published_date})" if s.published_date else ""
            lines.append(f"{i}. [{s.title}]({s.url}){date} — via {s.provider}")
        return "\n".join(lines)

    @staticmethod
    def _faq_from_brief(brief: ResearchBrief) -> list[tuple[str, str]]:
        """Build FAQ pairs from question-shaped search queries. Answers are
        honest placeholders the operator must fill from the brief — the
        pipeline never invents FAQ answers."""
        faqs: list[tuple[str, str]] = []
        for q in brief.queries:
            if q.strip().endswith("?") or q.lower().startswith(
                ("what ", "how ", "why ", "when ", "is ", "are ", "can ", "should ")
            ):
                faqs.append(
                    (
                        q if q.strip().endswith("?") else q.strip() + "?",
                        "_Answer from the research brief's section 6 — fill "
                        "from verified sources before publishing._",
                    )
                )
            if len(faqs) >= 5:
                break
        return faqs

    async def _spend_so_far(self) -> float:
        # Best-effort: not all enforcers expose spend; snapshot does.
        try:
            snap = await self._enforcer.snapshot()
            return float(snap.get("usd", 0.0))
        except Exception:
            return 0.0


class SearchNotConfigured(Exception):
    """No search provider key found for this user in the vault."""
