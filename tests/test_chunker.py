"""Tests for chunking: stable ids, token counts, metadata preservation."""

from __future__ import annotations

from guideline_gpt.config import Settings
from guideline_gpt.ingestion.chunker import (
    chunk_pages,
    count_tokens,
    make_chunk_id,
)
from guideline_gpt.types import DocumentMetadata


def _meta(page: int = 1) -> DocumentMetadata:
    return DocumentMetadata(
        source_path="/corpus/guide.pdf",
        source_name="guide",
        page_number=page,
    )


def test_count_tokens_nonzero() -> None:
    assert count_tokens("the quick brown fox") > 0
    assert count_tokens("") == 0


def test_chunk_id_is_stable_and_short() -> None:
    a = make_chunk_id("/corpus/guide.pdf", 1, "some text")
    b = make_chunk_id("/corpus/guide.pdf", 1, "some text")
    assert a == b
    assert len(a) == 16


def test_chunk_id_varies_with_inputs() -> None:
    base = make_chunk_id("/corpus/guide.pdf", 1, "text")
    assert base != make_chunk_id("/corpus/guide.pdf", 2, "text")
    assert base != make_chunk_id("/corpus/other.pdf", 1, "text")
    assert base != make_chunk_id("/corpus/guide.pdf", 1, "different")


def test_chunk_pages_preserves_metadata(settings: Settings) -> None:
    pages = [("Sentence one. Sentence two. Sentence three.", _meta(page=7))]
    chunks = chunk_pages(pages, settings)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata.page_number == 7
        assert chunk.metadata.source_name == "guide"
        assert chunk.token_count > 0
        assert chunk.chunk_id == make_chunk_id(
            chunk.metadata.source_path, chunk.metadata.page_number, chunk.text
        )


def test_long_text_splits_into_multiple_chunks(settings: Settings) -> None:
    cfg = settings.model_copy(update={"chunk_size": 32, "chunk_overlap": 4})
    long_text = " ".join(f"This is sentence number {i} about ventilation." for i in range(80))
    chunks = chunk_pages([(long_text, _meta())], cfg)
    assert len(chunks) > 1
    # Every chunk should respect the configured size (with small tokenizer slack).
    assert all(c.token_count <= cfg.chunk_size + cfg.chunk_overlap for c in chunks)


def test_blank_pages_produce_no_chunks(settings: Settings) -> None:
    assert chunk_pages([("   ", _meta())], settings) == []
