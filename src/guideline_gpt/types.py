"""Shared data types passed across module boundaries.

Every public function in the package takes and returns these dataclasses rather
than untyped dicts (see the engineering principles in the spec). The
:class:`QueryTrace` object is the contract between the pipeline and the UI: the
inspector view is simply a rendering of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

RetrievalSource = Literal["vector", "bm25", "hybrid", "rerank"]


@dataclass(frozen=True)
class DocumentMetadata:
    """Provenance for a chunk, traced back to its source PDF page."""

    source_path: str
    """Original PDF path."""
    source_name: str
    """Display name (filename without extension)."""
    page_number: int
    """1-indexed page the text came from."""
    section_heading: str | None = None
    """Nearest preceding section heading, if detected."""


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text with stable identity and provenance."""

    chunk_id: str
    """Stable hash of content + metadata (see ``ingestion.chunker``)."""
    text: str
    metadata: DocumentMetadata
    token_count: int


@dataclass(frozen=True)
class RetrievalHit:
    """A chunk returned by a retrieval stage, with its score and rank."""

    chunk: Chunk
    score: float
    source: RetrievalSource
    rank: int
    """1-indexed position within the stage that produced this hit."""


@dataclass
class QueryTrace:
    """Captures every stage of a query for the inspector UI.

    This object is mutable by design: each pipeline stage appends to it as it
    runs, so that even on partial failure the trace reflects what succeeded.
    """

    query: str
    timestamp: datetime
    vector_hits: list[RetrievalHit] = field(default_factory=list)
    bm25_hits: list[RetrievalHit] = field(default_factory=list)
    fused_hits: list[RetrievalHit] = field(default_factory=list)
    reranked_hits: list[RetrievalHit] = field(default_factory=list)
    prompt_assembled: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    llm_latency_ms: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    stage_errors: dict[str, str] = field(default_factory=dict)
    """Maps a stage name to an error message when that stage failed."""


@dataclass(frozen=True)
class QueryResponse:
    """The final result of a query: the answer, its sources, and the full trace."""

    answer: str
    citations: list[Chunk]
    trace: QueryTrace


@dataclass(frozen=True)
class IngestionReport:
    """Summary statistics produced by an ingestion run."""

    files: int
    pages: int
    chunks: int
    total_tokens: int
    elapsed_seconds: float
