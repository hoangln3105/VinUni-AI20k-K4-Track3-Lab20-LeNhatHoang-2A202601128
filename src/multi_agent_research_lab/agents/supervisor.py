"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"
_ROUTES = ("researcher", "analyst", "writer", "critic", DONE)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None, max_errors: int = 3) -> None:
        self._max_iterations = max_iterations or get_settings().max_iterations
        self._max_errors = max_errors

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history`."""

        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor_route", {"route": route, "iteration": state.iteration}
        )
        return state

    def decide(self, state: ResearchState) -> str:
        """Pure routing policy, kept separate from state mutation for easy unit testing."""

        if state.iteration >= self._max_iterations:
            return DONE
        if len(state.errors) >= self._max_errors:
            return DONE
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        if state.critic_notes is None:
            return "critic"
        return DONE
