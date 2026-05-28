"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from guideline_gpt.config import Settings


@pytest.fixture
def settings(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """A Settings instance isolated from the developer's real environment.

    API keys are stubbed and storage paths point at a temporary directory so
    tests never touch a real corpus or vector store.
    """
    # Prevent a developer's real .env / shell vars from leaking into tests.
    for var in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "CHROMA_PERSIST_DIR",
        "DOCUMENTS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = Settings(
        _env_file=None,  # type: ignore[call-arg]
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        chroma_persist_dir=tmp_path / "chroma",  # type: ignore[operator]
        documents_dir=tmp_path / "documents",  # type: ignore[operator]
    )
    yield cfg
