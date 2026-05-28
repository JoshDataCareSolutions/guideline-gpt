"""Tests for configuration loading and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from guideline_gpt.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.llm_provider == "anthropic"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.retrieval_top_k == 20
    assert settings.rerank_top_k == 5
    assert settings.embedding_model == "text-embedding-3-small"


def test_bm25_path_is_beside_vector_store(settings: Settings) -> None:
    assert settings.bm25_path == settings.chroma_persist_dir / "bm25.pkl"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    cfg = Settings(_env_file=None)  # type: ignore[call-arg]
    assert cfg.llm_provider == "openai"
    assert cfg.chunk_size == 256
    assert cfg.retrieval_top_k == 10


def test_paths_are_path_objects(settings: Settings) -> None:
    assert isinstance(settings.chroma_persist_dir, Path)
    assert isinstance(settings.documents_dir, Path)


def test_invalid_chunk_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "0")
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic raises ValidationError (a ValueError)
        Settings(_env_file=None)  # type: ignore[call-arg]
