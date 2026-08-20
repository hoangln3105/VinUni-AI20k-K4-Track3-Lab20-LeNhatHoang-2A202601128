"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics], failure_mode_notes: str | None = None
) -> str:
    """Render benchmark metrics to markdown, with a short comparison summary up top."""

    lines = ["# Benchmark Report", ""]

    if metrics:
        fastest = min(metrics, key=lambda item: item.latency_seconds)
        lines.append(f"- Fastest run: **{fastest.run_name}** ({fastest.latency_seconds:.2f}s)")

        scored = [item for item in metrics if item.quality_score is not None]
        if scored:
            best_quality = max(scored, key=lambda item: item.quality_score or 0.0)
            lines.append(
                f"- Highest quality: **{best_quality.run_name}** "
                f"({best_quality.quality_score:.1f}/10)"
            )

        failures = [item for item in metrics if item.failure_rate]
        if failures:
            names = ", ".join(item.run_name for item in failures)
            lines.append(f"- Runs with recovered/failed errors: {names}")
        lines.append("")

    lines += [
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    if failure_mode_notes:
        lines += ["", "## Failure mode encountered & fix", "", failure_mode_notes]

    return "\n".join(lines) + "\n"
