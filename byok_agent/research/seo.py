"""SEO pass — pure functions, no I/O, no model calls.

Implements the keyword-placement contract from the article-writing prompt:
front-loaded SEO title (<=60 chars), meta description (<=160 chars),
primary keyword in the first 100 words and in at least one H2, density
0.5%-1.5%, and a human-shaped FAQ section.
"""

from __future__ import annotations

import re


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def seo_title(primary_keyword: str, topic: str, max_len: int = 60) -> str:
    """SEO title with the primary keyword front-loaded, capped at 60 chars."""
    kw = primary_keyword.strip()
    base = f"{kw}: {topic.strip()}" if topic.strip().lower() not in kw.lower() else kw
    if len(base) <= max_len:
        return base
    # Truncate on a word boundary, keep the keyword intact.
    cut = base[: max_len - 1].rsplit(" ", 1)[0]
    return cut if kw.lower() in cut.lower() else base[: max_len - 3].rstrip() + "..."


def meta_description(summary: str, primary_keyword: str, max_len: int = 160) -> str:
    """Meta description capped at 160 chars, keyword included when it fits."""
    text = " ".join(summary.split())
    if primary_keyword.lower() not in text.lower():
        text = f"{primary_keyword}: {text}"
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 3].rsplit(" ", 1)[0]
    return cut + "..."


def keyword_density(text: str, keyword: str) -> float:
    """(occurrences / total words) * 100, case-insensitive substring match."""
    words = _words(text)
    if not words or not keyword.strip():
        return 0.0
    kw = keyword.strip().lower()
    count = sum(1 for _ in re.finditer(re.escape(kw), text.lower()))
    return (count / len(words)) * 100


def keyword_in_first_n_words(text: str, keyword: str, n: int = 100) -> bool:
    """Primary keyword appears (case-insensitive) within the first n words."""
    first = " ".join(_words(text)[:n])
    return keyword.strip().lower() in first.lower()


def h2_contains_keyword(headings: list[str], keyword: str) -> bool:
    """At least one H2 contains the keyword or a close variant (first 3 words)."""
    kw = keyword.strip().lower()
    variant = " ".join(kw.split()[:3])
    return any(variant in h.lower() for h in headings)


def faq_markdown(faqs: list[tuple[str, str]]) -> str:
    """Render a human-shaped FAQ: no question restated in the answer, varied
    answer lengths preserved as given."""
    parts = ["## Frequently Asked Questions", ""]
    for q, a in faqs:
        parts.append(f"**{q.strip()}**")
        parts.append("")
        parts.append(a.strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def assemble_article(
    *,
    title: str,
    intro: str,
    sections: list[tuple[str, str]],
    faq_md: str = "",
    sources_md: str = "",
) -> str:
    """Assemble the final Markdown article: H1, intro, H2 sections, FAQ,
    sources. Sections are (heading, body) pairs."""
    parts = [f"# {title.strip()}", "", intro.strip(), ""]
    for heading, body in sections:
        parts.append(f"## {heading.strip()}")
        parts.append("")
        parts.append(body.strip())
        parts.append("")
    if faq_md.strip():
        parts.append(faq_md.strip())
        parts.append("")
    if sources_md.strip():
        parts.append("## Sources")
        parts.append("")
        parts.append(sources_md.strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
