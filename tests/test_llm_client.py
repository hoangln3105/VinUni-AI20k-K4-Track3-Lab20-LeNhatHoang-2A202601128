from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import LLMClient


def test_mock_complete_without_api_key() -> None:
    # Explicitly inject a keyless Settings so this test is deterministic regardless of
    # whatever OPENAI_API_KEY happens to be configured in the developer's local .env.
    client = LLMClient(settings=Settings(openai_api_key=None))
    response = client.complete("You are a helpful assistant.", "Summarize: hello world")
    assert response.content
    assert response.input_tokens and response.input_tokens > 0
    assert response.output_tokens and response.output_tokens > 0
    assert response.cost_usd == 0.0
