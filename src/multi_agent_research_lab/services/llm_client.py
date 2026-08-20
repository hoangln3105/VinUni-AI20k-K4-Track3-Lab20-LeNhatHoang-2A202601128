"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Rough public pricing for gpt-4o-mini-class models, used only to estimate benchmark cost.
_PRICE_PER_1K_INPUT_USD = 0.00015
_PRICE_PER_1K_OUTPUT_USD = 0.0006


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client.

    Calls OpenAI's Chat Completions API when `OPENAI_API_KEY` is configured. Otherwise falls
    back to a deterministic offline mock so agents, tests, and demos keep working without keys.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._model = settings.openai_model
        self._client = None
        if settings.openai_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Retries transient provider errors up to 3 attempts."""

        if self._client is None:
            return self._mock_complete(system_prompt, user_prompt)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost_usd = None
        if input_tokens is not None and output_tokens is not None:
            cost_usd = (
                input_tokens / 1000 * _PRICE_PER_1K_INPUT_USD
                + output_tokens / 1000 * _PRICE_PER_1K_OUTPUT_USD
            )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _mock_complete(system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic offline fallback used when no OPENAI_API_KEY is configured."""

        logger.info("LLMClient running in offline mock mode (no OPENAI_API_KEY set).")
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        preview = " ".join(lines[:3])[:400]
        content = f"(offline mock, role={system_prompt.split('.')[0][:60]!r}) {preview}"
        input_tokens = len(system_prompt.split()) + len(user_prompt.split())
        output_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )
