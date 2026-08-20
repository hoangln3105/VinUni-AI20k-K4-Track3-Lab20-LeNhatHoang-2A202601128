"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a research analyst. Read the provided sources and write concise, well-cited "
    "research notes. Cite sources inline using their bracket number, e.g. [1]."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self._search = search_client or SearchClient()
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self._search.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources

        source_list = "\n".join(
            f"[{i}] {source.title}: {source.snippet}" for i, source in enumerate(sources, start=1)
        )
        user_prompt = (
            f"Research query: {state.request.query}\n\n"
            f"Sources:\n{source_list or '(no sources found)'}\n\n"
            "Write 4-6 bullet points of research notes covering the query, citing sources "
            "inline as [n]."
        )
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
