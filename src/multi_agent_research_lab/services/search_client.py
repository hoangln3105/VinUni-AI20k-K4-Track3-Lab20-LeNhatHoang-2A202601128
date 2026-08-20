"""Search client abstraction for ResearcherAgent."""

import json
import logging
import urllib.request
from urllib.error import URLError

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchClient:
    """Provider-agnostic search client.

    Calls Tavily's REST API when `TAVILY_API_KEY` is configured. Otherwise falls back to a
    deterministic offline mock corpus so ResearcherAgent keeps working without a key.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._api_key = (settings or get_settings()).tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self._api_key:
            try:
                return self._search_tavily(query, max_results)
            except (URLError, TimeoutError, ValueError, OSError) as exc:
                logger.warning("Tavily search failed (%s); falling back to offline mock.", exc)
        return self._mock_search(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            _TAVILY_ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        results = data.get("results", [])[:max_results]
        return [
            SourceDocument(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item.get("url"),
                snippet=(item.get("content") or "")[:500],
                metadata={"score": item.get("score")},
            )
            for item in results
        ]

    @staticmethod
    def _mock_search(query: str, max_results: int) -> list[SourceDocument]:
        """Deterministic offline fallback used when no TAVILY_API_KEY is configured."""

        logger.info("SearchClient running in offline mock mode (no TAVILY_API_KEY set).")
        return [
            SourceDocument(
                title=f"Offline mock source {i} for '{query}'",
                url=None,
                snippet=(
                    f"(offline mock) Synthesized background note #{i} relevant to '{query}'. "
                    "Configure TAVILY_API_KEY for real web search results."
                ),
                metadata={"mock": True, "rank": i},
            )
            for i in range(1, max_results + 1)
        ]
