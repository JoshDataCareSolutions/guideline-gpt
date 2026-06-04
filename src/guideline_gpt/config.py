"""Centralized, type-safe configuration loaded from environment variables.

Anything tunable in the system lives here. No business-logic module should read
``os.environ`` directly or bury magic constants in functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider selection ------------------------------------------
    llm_provider: LLMProvider = "anthropic"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # OpenAI key is always required (embeddings); also used as LLM when selected.
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"

    # --- Reranker ---------------------------------------------------------
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Ingestion cleaning -----------------------------------------------
    # A line repeated on at least this fraction of a document's pages is treated
    # as a running header/footer (e.g. copyright notices) and stripped before
    # chunking. Set to 1.0 to effectively disable.
    boilerplate_page_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)

    # --- Storage / paths --------------------------------------------------
    chroma_persist_dir: Path = Path("./chroma_db")
    documents_dir: Path = Path("./documents")
    collection_name: str = "guidelines"

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def bm25_path(self) -> Path:
        """Filesystem path of the persisted BM25 index, beside the vector store."""
        return self.chroma_persist_dir / "bm25.pkl"


def get_settings() -> Settings:
    """Construct a :class:`Settings` instance from the current environment."""
    return Settings()
