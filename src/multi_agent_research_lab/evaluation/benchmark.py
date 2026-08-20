"""Benchmark harness for single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _estimate_cost_usd(state: ResearchState) -> float | None:
    costs = [
        cost
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    return round(sum(costs), 6) if costs else None


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.sources or not state.final_answer:
        return None
    cited = sum(
        1 for i in range(1, len(state.sources) + 1) if f"[{i}]" in state.final_answer
    )
    return cited / len(state.sources)


def _quality_score(state: ResearchState) -> float | None:
    """Heuristic 0-10 proxy from answer length and citation coverage.

    TODO(student, optional): replace with real scores collected via
    `docs/peer_review_rubric.md` once peer review has been run.
    """

    if not state.final_answer:
        return 0.0
    score = 4.0
    score += min(3.0, len(state.final_answer.split()) / 150)
    coverage = _citation_coverage(state)
    if coverage:
        score += coverage * 3.0
    return round(min(score, 10.0), 1)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)`, measuring latency, cost, quality, citation coverage, and failures."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a runner failure is itself a benchmark result
        latency = perf_counter() - started
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"runner raised: {exc}",
        )
        return state, metrics

    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost_usd(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes=f"{len(state.errors)} recovered error(s)" if state.errors else "",
    )
    return state, metrics
