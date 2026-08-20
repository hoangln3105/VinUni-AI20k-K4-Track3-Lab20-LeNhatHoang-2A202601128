"""Tracing hooks.

`trace_span` records local spans into `ResearchState.trace`. `configure_tracing` additionally
wires up LangSmith when configured: LangChain/LangGraph auto-instrument every run once the
`LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY` env vars are set, so no per-call SDK code is needed.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Enable external trace providers from settings, if configured."""

    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        logger.info("LangSmith tracing enabled for project '%s'.", settings.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    logger.debug("span start: %s %s", name, attributes or {})
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("span end: %s (%.3fs)", name, span["duration_seconds"])
