"""Critic agent: fact-check / citation-coverage guard for the final answer."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Validates the final answer and appends findings; does not call an LLM."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate `state.final_answer` and populate `state.critic_notes`."""

        if not state.final_answer:
            raise AgentExecutionError("CriticAgent requires a final_answer before running.")

        findings: list[str] = []

        total_sources = len(state.sources)
        if total_sources:
            cited = sum(
                1 for i in range(1, total_sources + 1) if f"[{i}]" in state.final_answer
            )
            coverage = cited / total_sources
            findings.append(
                f"Citation coverage: {coverage:.0%} ({cited}/{total_sources} sources referenced)."
            )
            if coverage < 0.5:
                findings.append("Warning: final answer under-cites the gathered sources.")
        else:
            coverage = 0.0
            findings.append("No sources were gathered; citation coverage is undefined.")

        word_count = len(state.final_answer.split())
        if word_count < 20:
            findings.append(f"Warning: final answer looks too short ({word_count} words).")

        state.critic_notes = "\n".join(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=state.critic_notes,
                metadata={"citation_coverage": coverage, "word_count": word_count},
            )
        )
        return state
