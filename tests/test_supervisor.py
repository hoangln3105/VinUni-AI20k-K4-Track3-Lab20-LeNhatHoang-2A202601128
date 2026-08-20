from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_first() -> None:
    assert SupervisorAgent().decide(_state()) == "researcher"


def test_routes_through_analyst_then_writer_then_critic() -> None:
    state = _state()
    state.research_notes = "notes"
    assert SupervisorAgent().decide(state) == "analyst"

    state.analysis_notes = "analysis"
    assert SupervisorAgent().decide(state) == "writer"

    state.final_answer = "answer"
    assert SupervisorAgent().decide(state) == "critic"

    state.critic_notes = "critic findings"
    assert SupervisorAgent().decide(state) == "done"


def test_stops_at_max_iterations() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=2)
    supervisor.run(state)
    assert state.route_history[-1] == "researcher"
    assert state.iteration == 1
    supervisor.run(state)
    assert state.route_history[-1] == "researcher"
    assert state.iteration == 2
    supervisor.run(state)
    assert state.route_history[-1] == "done"
    assert state.iteration == 3


def test_stops_after_too_many_errors() -> None:
    state = _state()
    state.errors = ["boom", "boom", "boom"]
    assert SupervisorAgent().decide(state) == "done"
