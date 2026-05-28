"""Tests for the provider-agnostic LLM client and factory (mocked SDKs)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from guideline_gpt.config import Settings
from guideline_gpt.generation.llm_client import (
    AnthropicClient,
    OpenAIClient,
    get_llm_client,
)


def test_anthropic_client_normalizes_response() -> None:
    fake = MagicMock()
    fake.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="PEEP guidance [1].")],
        usage=SimpleNamespace(input_tokens=42, output_tokens=8),
        model="claude-haiku-4-5-20251001",
    )
    client = AnthropicClient("key", "claude-haiku-4-5-20251001", client=fake)

    result = client.complete("system", "user")

    assert result.text == "PEEP guidance [1]."
    assert result.input_tokens == 42
    assert result.output_tokens == 8
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.latency_ms >= 0
    # The system prompt is passed as a top-level arg, not a message.
    _, kwargs = fake.messages.create.call_args
    assert kwargs["system"] == "system"
    assert kwargs["messages"] == [{"role": "user", "content": "user"}]


def test_openai_client_normalizes_response() -> None:
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Septic shock [2]."))],
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5),
        model="gpt-4o-mini",
    )
    client = OpenAIClient("key", "gpt-4o-mini", client=fake)

    result = client.complete("system", "user")

    assert result.text == "Septic shock [2]."
    assert result.input_tokens == 30
    assert result.output_tokens == 5
    assert result.model == "gpt-4o-mini"
    _, kwargs = fake.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "system"}
    assert kwargs["messages"][1] == {"role": "user", "content": "user"}


def test_openai_handles_missing_usage_and_content() -> None:
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=None,
        model="gpt-4o-mini",
    )
    result = OpenAIClient("key", "gpt-4o-mini", client=fake).complete("s", "u")
    assert result.text == ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_factory_selects_anthropic(settings: Settings) -> None:
    client = get_llm_client(settings)
    assert isinstance(client, AnthropicClient)


def test_factory_selects_openai(settings: Settings) -> None:
    cfg = settings.model_copy(update={"llm_provider": "openai"})
    assert isinstance(get_llm_client(cfg), OpenAIClient)


def test_factory_requires_anthropic_key(settings: Settings) -> None:
    cfg = settings.model_copy(update={"anthropic_api_key": None})
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm_client(cfg)


def test_factory_requires_openai_key(settings: Settings) -> None:
    cfg = settings.model_copy(update={"llm_provider": "openai", "openai_api_key": None})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm_client(cfg)
