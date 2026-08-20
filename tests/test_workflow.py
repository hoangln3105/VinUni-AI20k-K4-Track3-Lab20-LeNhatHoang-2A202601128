import pytest

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


@pytest.fixture(autouse=True)
def _force_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline mock mode for every agent in the graph.

    Agents construct their own LLMClient/SearchClient via `get_settings()`, so a real key in
    the developer's local `.env` would otherwise make this test hit live APIs.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_multi_agent_workflow_runs_end_to_end_offline() -> None:
    request = ResearchQuery(query="Explain multi-agent systems for engineers")
    state = ResearchState(request=request)

    result = MultiAgentWorkflow().run(state)

    assert result.research_notes
    assert result.analysis_notes
    assert result.final_answer
    assert result.critic_notes
    assert result.route_history == ["researcher", "analyst", "writer", "critic", "done"]
    assert result.iteration == 5
    assert not result.errors
