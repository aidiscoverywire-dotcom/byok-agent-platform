"""Research & articles pipeline (Phase 3).

STORM-inspired stages (plan -> gather -> outline -> draft -> verify ->
SEO) over pluggable search providers, with an audited verification gate:
the verifier NEVER sees the draft stage's evidence set (disjoint by
construction), contradicted claims always block publish, and the
revise -> verify loop has a hard cap.
"""

from byok_agent.research.nli import (
    CrossEncoderNLI,
    LLMJudgeNLI,
    NLIModel,
    NLIVerdict,
    chunk_text,
)
from byok_agent.research.pipeline import (
    ArticleResult,
    ResearchBrief,
    ResearchPipeline,
)
from byok_agent.research.search import (
    BraveProvider,
    ExaProvider,
    SEARCH_PRICE_USD,
    SearchProvider,
    SearchResult,
    TavilyProvider,
    dedupe_results,
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
    AtomicClaim,
    ClaimCheck,
    EvidenceChunk,
    SectionVerification,
    VerificationFailed,
    VerificationGate,
)

__all__ = [
    "ArticleResult",
    "AtomicClaim",
    "BraveProvider",
    "ClaimCheck",
    "CrossEncoderNLI",
    "EvidenceChunk",
    "ExaProvider",
    "LLMJudgeNLI",
    "NLIModel",
    "NLIVerdict",
    "ResearchBrief",
    "ResearchPipeline",
    "SEARCH_PRICE_USD",
    "SearchProvider",
    "SearchResult",
    "SectionVerification",
    "TavilyProvider",
    "VerificationFailed",
    "VerificationGate",
    "assemble_article",
    "chunk_text",
    "dedupe_results",
    "faq_markdown",
    "keyword_density",
    "keyword_in_first_n_words",
    "meta_description",
    "seo_title",
]
