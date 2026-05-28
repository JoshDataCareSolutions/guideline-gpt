"""Provider-agnostic LLM access.

All LLM calls in the system go through the :class:`LLMClient` protocol. Two
concrete implementations (:class:`AnthropicClient`, :class:`OpenAIClient`) are
selected by :func:`get_llm_client` based on ``settings.llm_provider``. No other
module imports a provider SDK directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from guideline_gpt.config import Settings

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

# Anthropic requires an explicit max output token budget; OpenAI defaults are fine
# but we cap both for predictable cost.
DEFAULT_MAX_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class CompletionResult:
    """The result of a single LLM completion, with usage and timing."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str


class LLMClient(Protocol):
    """A minimal, provider-agnostic chat completion interface."""

    def complete(self, system: str, user: str) -> CompletionResult:
        """Generate a completion for a system + user prompt pair."""
        ...


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, *, client: Anthropic | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model

    def complete(self, system: str, user: str) -> CompletionResult:
        """Call the Anthropic Messages API and return a normalized result."""
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = "".join(block.text for block in response.content if block.type == "text")
        return CompletionResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=response.model,
        )


class OpenAIClient:
    """LLMClient backed by the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str, *, client: OpenAI | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model

    def complete(self, system: str, user: str) -> CompletionResult:
        """Call the OpenAI Chat Completions API and return a normalized result."""
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = response.usage
        return CompletionResult(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=response.model,
        )


def get_llm_client(settings: Settings) -> LLMClient:
    """Construct the configured LLM client.

    Args:
        settings: Configuration providing the provider, model, and API keys.

    Returns:
        A concrete :class:`LLMClient`.

    Raises:
        ValueError: If the required API key for the selected provider is missing.
    """
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    return OpenAIClient(settings.openai_api_key, settings.openai_model)
