"""Pluggable search providers for the research pipeline.

Priority order: Tavily (content-in-response, primary) -> Exa (secondary)
-> Brave (bulk fallback). API keys live in the Phase 0 vault as key
records with ``provider="tavily" / "exa" / "brave"``, tenant-scoped by
owner — the same BYOK model as LLM keys. Every search call is metered
in the Phase 0 ledger; prices are research-derived ESTIMATES
(Oct 2026), never billing truth.

All HTTP goes through ``httpx``. Providers accept an injectable
``http_client`` (anything with an async ``post``/``get`` returning an
object with ``status_code`` and ``json()``) so tests never touch the
network; a real ``httpx.AsyncClient`` is created only when none is
supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Estimated price per SEARCH CALL in USD (research-derived, Oct 2026).
# Tavily: $8/1k basic, $16/1k advanced. Exa: $7/1k auto. Brave: $5/1k.
# Flat per call — result count does not change the price tier.
SEARCH_PRICE_USD: dict[str, dict[str, float]] = {
    "tavily": {"basic": 0.008, "advanced": 0.016},
    "exa": {"auto": 0.007},
    "brave": {"search": 0.005},
}

SEARCH_PROVIDER_NAMES = ("tavily", "exa", "brave")


@dataclass
class SearchResult:
    """One normalized search hit."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""  # full extracted text, when the provider returns it
    published_date: str = ""  # ISO date when the provider supplies one
    provider: str = ""  # which provider returned this hit


@runtime_checkable
class SearchProvider(Protocol):
    """Async search contract every provider must satisfy."""

    name: str

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        api_key: str,
        http_client: Any = None,
    ) -> list[SearchResult]:
        """Run one query. ``api_key`` is plaintext at the call boundary only."""
        ...

    def estimate_cost_usd(self, max_results: int = 10) -> float:
        """Estimated cost of ONE search call (flat per call)."""
        ...


def _norm_url(url: str) -> str:
    """Normalize a URL for dedupe: lowercase host AND path, strip trailing
    slash and common tracking query params.

    Note: URL paths are technically case-sensitive per RFC, but for
    research dedupe, ``/X/`` vs ``/x`` almost always point at the same
    article (and trailing-slash variants always do). This is a dedupe
    heuristic, not a fetch canonicalizer — the original URL is preserved
    on the SearchResult.
    """
    try:
        p = urlparse(url.strip())
        host = p.netloc.lower()
        path = (p.path.rstrip("/") or "/").lower()
        return f"{p.scheme}://{host}{path}"
    except Exception:
        return url.strip().lower()


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Drop duplicate URLs, keeping the first (highest-ranked) occurrence."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        key = _norm_url(r.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


async def find_search_key(vault: Any, owner: str, provider_name: str) -> Any | None:
    """Return the active vault key record for a search provider, or None.

    Tenant-scoped: only keys owned by ``owner`` are visible (the vault
    itself enforces this — no cross-tenant existence oracle).
    """
    for record in await vault.list_keys(owner=owner):
        if record.provider == provider_name and record.status.value == "active":
            return record
    return None


class _HttpMixin:
    """Shared httpx plumbing: injectable client, lazy real one otherwise."""

    async def _post(self, url: str, *, headers: dict, payload: dict, http_client: Any):
        if http_client is None:
            import httpx  # lazy: keeps import cost off the CLI path

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
        else:
            resp = await http_client.post(url, headers=headers, json=payload)
        return resp

    async def _get(self, url: str, *, headers: dict, params: dict, http_client: Any):
        if http_client is None:
            import httpx  # lazy

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers, params=params)
        else:
            resp = await http_client.get(url, headers=headers, params=params)
        return resp

    @staticmethod
    def _raise_for_status(resp: Any, provider: str) -> None:
        code = getattr(resp, "status_code", 200)
        if code >= 400:
            raise SearchError(f"{provider} search failed: HTTP {code}")


class SearchError(Exception):
    """A search provider call failed (network, auth, or bad response)."""


class TavilyProvider(_HttpMixin):
    """Tavily — primary provider. ``search_depth="basic"`` is the cheap
    default; ``"advanced"`` costs 2x and is reserved for deep runs."""

    name = "tavily"

    def __init__(self, *, search_depth: str = "basic", http_client: Any = None) -> None:
        if search_depth not in ("basic", "advanced"):
            raise ValueError("search_depth must be 'basic' or 'advanced'")
        self._depth = search_depth
        self._http_client = http_client

    def estimate_cost_usd(self, max_results: int = 10) -> float:
        return SEARCH_PRICE_USD["tavily"][self._depth]

    async def search(
        self, query: str, max_results: int, *, api_key: str, http_client: Any = None
    ) -> list[SearchResult]:
        client = http_client or self._http_client
        resp = await self._post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            payload={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": self._depth,
                "include_answer": False,
            },
            http_client=client,
        )
        self._raise_for_status(resp, "tavily")
        data = resp.json() if callable(getattr(resp, "json", None)) else {}
        out: list[SearchResult] = []
        for hit in (data.get("results") or [])[:max_results]:
            out.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=hit.get("url", ""),
                    snippet=hit.get("content", "")[:500],
                    content=hit.get("content", ""),
                    published_date=hit.get("published_date", ""),
                    provider="tavily",
                )
            )
        return out


class ExaProvider(_HttpMixin):
    """Exa — secondary provider. ``contents.text`` returns extracted page
    text in the same call (no second fetch needed)."""

    name = "exa"

    def __init__(self, *, http_client: Any = None) -> None:
        self._http_client = http_client

    def estimate_cost_usd(self, max_results: int = 10) -> float:
        return SEARCH_PRICE_USD["exa"]["auto"]

    async def search(
        self, query: str, max_results: int, *, api_key: str, http_client: Any = None
    ) -> list[SearchResult]:
        client = http_client or self._http_client
        resp = await self._post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            payload={
                "query": query,
                "numResults": max_results,
                "contents": {"text": True},
            },
            http_client=client,
        )
        self._raise_for_status(resp, "exa")
        data = resp.json() if callable(getattr(resp, "json", None)) else {}
        out: list[SearchResult] = []
        for hit in (data.get("results") or [])[:max_results]:
            text = hit.get("text", "")
            out.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=hit.get("url", ""),
                    snippet=text[:500],
                    content=text,
                    published_date=hit.get("publishedDate", ""),
                    provider="exa",
                )
            )
        return out


class BraveProvider(_HttpMixin):
    """Brave Search — bulk fallback. Returns snippets only (no extracted
    page text); the pipeline treats snippet-only hits as lower-weight
    evidence."""

    name = "brave"

    def __init__(self, *, http_client: Any = None) -> None:
        self._http_client = http_client

    def estimate_cost_usd(self, max_results: int = 10) -> float:
        return SEARCH_PRICE_USD["brave"]["search"]

    async def search(
        self, query: str, max_results: int, *, api_key: str, http_client: Any = None
    ) -> list[SearchResult]:
        client = http_client or self._http_client
        resp = await self._get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": min(max_results, 20)},
            http_client=client,
        )
        self._raise_for_status(resp, "brave")
        data = resp.json() if callable(getattr(resp, "json", None)) else {}
        out: list[SearchResult] = []
        for hit in ((data.get("web") or {}).get("results") or [])[:max_results]:
            out.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=hit.get("url", ""),
                    snippet=hit.get("description", ""),
                    content="",  # Brave returns snippets only
                    published_date=hit.get("page_age", ""),
                    provider="brave",
                )
            )
        return out
