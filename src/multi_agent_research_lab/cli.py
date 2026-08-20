"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a single all-purpose research assistant. Research, analyze, and write a final "
    "answer to the user's query in one pass, citing sources inline as [n] when known."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(request: ResearchQuery) -> ResearchState:
    state = ResearchState(request=request)
    response = LLMClient().complete(
        _BASELINE_SYSTEM_PROMPT, f"Query: {request.query}\nAudience: {request.audience}"
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "mode": "single-agent-baseline",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def _load_benchmark_queries() -> list[str]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "lab_default.yaml"
    default_queries = ["Research GraphRAG state-of-the-art and write a 500-word summary"]
    if not config_path.exists():
        return default_queries
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    queries = data.get("benchmark", {}).get("queries")
    return queries or default_queries


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline: one LLM call does research, analysis, and writing."""

    _init()
    state = _run_baseline(_parse_query(query))
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer -> Critic."""

    _init()
    state = ResearchState(request=_parse_query(query))
    result = MultiAgentWorkflow().run(state)
    console.print(
        Panel.fit(result.final_answer or "(no final answer)", title="Multi-Agent Final Answer")
    )
    console.print(f"Route history: {' -> '.join(result.route_history)}")
    if result.critic_notes:
        console.print(Panel.fit(result.critic_notes, title="Critic Notes"))
    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="Recovered errors", style="yellow"))
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    output: Annotated[
        str, typer.Option("--output", "-o", help="Report path relative to reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline vs multi-agent over the configured queries and write a markdown report."""

    _init()
    queries = _load_benchmark_queries()
    metrics = []

    for i, query in enumerate(queries, start=1):
        _, baseline_metrics = run_benchmark(
            f"baseline-{i}", query, lambda q: _run_baseline(ResearchQuery(query=q))
        )
        metrics.append(baseline_metrics)

        _, multi_metrics = run_benchmark(
            f"multi-agent-{i}",
            query,
            lambda q: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=q))),
        )
        metrics.append(multi_metrics)

    report = render_markdown_report(metrics)
    path = LocalArtifactStore().write_text(output, report)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark"))
    console.print(report)


if __name__ == "__main__":
    app()
