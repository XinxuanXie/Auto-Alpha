"""Key-free live web search for Research Agents."""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
from typing import Any, Callable, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WebResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^WEB-\d{2}$")
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    snippet: str = Field(min_length=1, max_length=2_000)


class WebResearchProvider(Protocol):
    def search(self, query: str | Sequence[str]) -> list[WebResearchSource]:
        """Return current web sources for one specialist query."""


class DDGSWebResearchProvider:
    def __init__(
        self,
        max_results: int = 5,
        timeout_seconds: int = 10,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory

    def search(self, query: str | Sequence[str]) -> list[WebResearchSource]:
        queries = [query] if isinstance(query, str) else list(query)
        normalized_queries = [
            " ".join(candidate.split()) for candidate in queries if candidate.strip()
        ]
        if not normalized_queries:
            raise ValueError("web research query cannot be empty")
        factory = self.client_factory
        if factory is None:
            from ddgs import DDGS

            factory = DDGS
        sources = []
        seen_urls = set()
        errors = []
        client = factory(timeout=self.timeout_seconds)
        for normalized_query in normalized_queries:
            try:
                raw_results = client.text(
                    normalized_query,
                    max_results=self.max_results,
                )
            except Exception as error:
                errors.append(str(error))
                continue
            for result in raw_results or []:
                title = " ".join(str(result.get("title") or "").split())
                url = str(
                    result.get("href") or result.get("url") or ""
                ).strip()
                snippet = " ".join(
                    str(
                        result.get("body") or result.get("snippet") or ""
                    ).split()
                )
                if not title or not url or not snippet or url in seen_urls:
                    continue
                try:
                    source = WebResearchSource(
                        source_id=f"WEB-{len(sources) + 1:02d}",
                        title=title,
                        url=url,
                        snippet=snippet,
                    )
                except ValueError:
                    continue
                sources.append(source)
                seen_urls.add(url)
                if len(sources) >= self.max_results:
                    break
            if len(sources) >= self.max_results:
                break
        if not sources:
            detail = f": {'; '.join(errors)}" if errors else ""
            raise RuntimeError(f"web research returned no usable sources{detail}")
        return sources


class CrossrefWebResearchProvider:
    endpoint = "https://api.crossref.org/works"

    def __init__(
        self,
        max_results: int = 5,
        timeout_seconds: int = 10,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def search(self, query: str | Sequence[str]) -> list[WebResearchSource]:
        queries = [query] if isinstance(query, str) else list(query)
        sources = []
        seen_urls = set()
        errors = []
        for candidate in queries:
            normalized_query = " ".join(candidate.split())
            if not normalized_query:
                continue
            parameters = urlencode(
                {
                    "query.bibliographic": normalized_query,
                    "rows": self.max_results,
                    "select": "title,URL,DOI,abstract,container-title",
                }
            )
            request = Request(
                f"{self.endpoint}?{parameters}",
                headers={"User-Agent": "AutoAlpha-Agent-Harness/1.0"},
            )
            try:
                with self.opener(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as error:
                errors.append(str(error))
                continue
            items = payload.get("message", {}).get("items", [])
            for item in items:
                title_values = item.get("title") or []
                title = " ".join(str(title_values[0]).split()) if title_values else ""
                url = str(item.get("URL") or "").strip()
                if not url and item.get("DOI"):
                    url = f"https://doi.org/{item['DOI']}"
                abstract = re.sub(
                    r"<[^>]+>",
                    " ",
                    str(item.get("abstract") or ""),
                )
                snippet = " ".join(abstract.split())
                if not snippet:
                    containers = item.get("container-title") or []
                    container = str(containers[0]).strip() if containers else "Crossref"
                    snippet = f"Scholarly work indexed by Crossref in {container}: {title}"
                if not title or not url or url in seen_urls:
                    continue
                try:
                    source = WebResearchSource(
                        source_id=f"WEB-{len(sources) + 1:02d}",
                        title=title,
                        url=url,
                        snippet=snippet,
                    )
                except ValueError:
                    continue
                sources.append(source)
                seen_urls.add(url)
                if len(sources) >= self.max_results:
                    break
            if len(sources) >= self.max_results:
                break
        if not sources:
            detail = f": {'; '.join(errors)}" if errors else ""
            raise RuntimeError(f"Crossref returned no usable sources{detail}")
        return sources


class KeyFreeWebResearchProvider:
    def __init__(
        self,
        max_results: int = 5,
        timeout_seconds: int = 10,
        *,
        providers: Sequence[WebResearchProvider] | None = None,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.max_results = max_results
        self.providers = tuple(
            providers
            or (
                CrossrefWebResearchProvider(max_results, timeout_seconds),
                DDGSWebResearchProvider(max_results, timeout_seconds),
            )
        )

    def search(self, query: str | Sequence[str]) -> list[WebResearchSource]:
        errors = []
        for provider in self.providers:
            try:
                sources = provider.search(query)
            except Exception as error:
                errors.append(f"{type(provider).__name__}: {error}")
                continue
            return [
                source.model_copy(update={"source_id": f"WEB-{index:02d}"})
                for index, source in enumerate(
                    sources[: self.max_results],
                    start=1,
                )
            ]
        raise RuntimeError("all web research providers failed: " + "; ".join(errors))


def build_strategy_search_query(strategy_key: str | None) -> str:
    return build_strategy_search_queries(strategy_key)[0]


def build_strategy_search_queries(strategy_key: str | None) -> tuple[str, ...]:
    strategy_queries = {
        "momentum": (
            "cross-sectional stock return momentum volume trend empirical research",
            "stock momentum trading volume factor",
        ),
        "reversal": (
            "short-term stock reversal abnormal volume empirical research",
            "intraday range volume stock return reversal",
        ),
        "liquidity_volatility": (
            "stock liquidity volatility risk premium empirical research",
            "liquidity volatility cross-sectional stock returns",
        ),
        "price_volume": (
            "trading volume stock price return relation",
            "abnormal trading volume return continuation reversal",
        ),
    }
    return strategy_queries.get(
        strategy_key,
        (
            "OHLCV cross-sectional stock return factor empirical research",
            "technical stock return factors price volume",
        ),
    )
