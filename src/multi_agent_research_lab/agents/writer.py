"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a technical writer. Synthesize a clear, well-organized final answer for {audience}, "
    "referencing sources inline as [n] where relevant."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.analysis_notes:
            raise AgentExecutionError("WriterAgent requires analysis_notes before running.")

        citations = "\n".join(
            f"[{i}] {source.title} ({source.url or 'no url'})"
            for i, source in enumerate(state.sources, start=1)
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Analysis notes:\n{state.analysis_notes}\n\n"
            f"Available sources:\n{citations or '(none)'}\n\n"
            "Write the final answer to the query."
        )
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(audience=state.request.audience)
        response = self._llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
