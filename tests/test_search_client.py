from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.search_client import SearchClient


def test_mock_search_without_api_key() -> None:
    # Explicitly inject a keyless Settings so this test is deterministic regardless of
    # whatever TAVILY_API_KEY happens to be configured in the developer's local .env.
    client = SearchClient(settings=Settings(tavily_api_key=None))
    results = client.search("multi-agent systems", max_results=3)
    assert len(results) == 3
    assert all(result.snippet for result in results)
