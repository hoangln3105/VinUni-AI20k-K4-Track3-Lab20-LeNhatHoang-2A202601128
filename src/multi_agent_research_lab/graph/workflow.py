"""LangGraph workflow: Supervisor routes to Researcher, Analyst, Writer, and Critic."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tenacity import Retrying, stop_after_attempt, wait_fixed

from multi_agent_research_lab.agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Guardrail: each failing worker gets one retry before the fallback text below is recorded.
_WORKER_RETRY_ATTEMPTS = 2


def _run_worker(
    agent: BaseAgent, state: ResearchState, fallback_field: str, fallback_text: str
) -> ResearchState:
    """Run `agent` with retry, recording a trace span and falling back on repeated failure."""

    with trace_span(agent.name, {"iteration": state.iteration}) as span:
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(_WORKER_RETRY_ATTEMPTS),
                wait=wait_fixed(0.2),
                reraise=True,
            ):
                with attempt:
                    state = agent.run(state)
        except Exception as exc:  # noqa: BLE001 - guardrail: a failing worker must not crash the graph
            logger.warning("%s failed after retries: %s", agent.name, exc)
            state.errors.append(f"{agent.name} failed: {exc}")
            if getattr(state, fallback_field) is None:
                setattr(state, fallback_field, fallback_text)
    state.add_trace_event(f"{agent.name}_done", {"duration_seconds": span["duration_seconds"]})
    return state


def _supervisor_node(state: ResearchState) -> ResearchState:
    return SupervisorAgent().run(state)


def _researcher_node(state: ResearchState) -> ResearchState:
    return _run_worker(
        ResearcherAgent(),
        state,
        "research_notes",
        "Researcher unavailable after retries; continuing without fresh sources.",
    )


def _analyst_node(state: ResearchState) -> ResearchState:
    return _run_worker(
        AnalystAgent(),
        state,
        "analysis_notes",
        "Analyst unavailable after retries; continuing with raw research notes only.",
    )


def _writer_node(state: ResearchState) -> ResearchState:
    return _run_worker(
        WriterAgent(),
        state,
        "final_answer",
        "Writer unavailable after retries; final answer could not be synthesized.",
    )


def _critic_node(state: ResearchState) -> ResearchState:
    return _run_worker(
        CriticAgent(),
        state,
        "critic_notes",
        "Critic unavailable after retries; final answer was not independently validated.",
    )


def _route_after_supervisor(state: ResearchState) -> str:
    return state.route_history[-1] if state.route_history else DONE


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
        """Create the compiled LangGraph graph: supervisor fans out to worker nodes and back."""

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", _supervisor_node)
        graph.add_node("researcher", _researcher_node)
        graph.add_node("analyst", _analyst_node)
        graph.add_node("writer", _writer_node)
        graph.add_node("critic", _critic_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                DONE: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and convert the result back to `ResearchState`."""

        settings = get_settings()
        compiled = self.build()
        result = compiled.invoke(
            state, config={"recursion_limit": settings.max_iterations * 4 + 10}
        )
        return ResearchState.model_validate(result)
