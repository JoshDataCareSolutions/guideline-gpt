"""Tests for BM25 tokenization, build/save/load, and search (mocked Chroma)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from guideline_gpt.config import Settings
from guideline_gpt.retrieval import bm25 as bm25_module
from guideline_gpt.retrieval.bm25 import (
    BM25Retriever,
    build_bm25,
    save_bm25,
    tokenize,
)
from guideline_gpt.types import Chunk, DocumentMetadata


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=DocumentMetadata(
            source_path=f"/c/{chunk_id}.pdf",
            source_name=chunk_id,
            page_number=1,
        ),
        token_count=len(text.split()),
    )


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("PEEP 10-15 cmH2O.") == ["peep", "10", "15", "cmh2o"]
    assert tokenize("ARDS!!!") == ["ards"]
    assert tokenize("") == []


# BM25Okapi's IDF can be 0 on tiny corpora (e.g., N=2, n=1 -> log(1.5/1.5)=0);
# tests use a larger corpus so ranking is exercised meaningfully.
def _corpus() -> list[Chunk]:
    return [
        _chunk("a", "PEEP higher strategy for moderate ARDS ventilation"),
        _chunk("b", "COPD bronchodilators and corticosteroids"),
        _chunk("c", "Sepsis fluid resuscitation and antibiotics"),
        _chunk("d", "Pulmonary embolism diagnosis and treatment"),
        _chunk("e", "Asthma controller medications"),
    ]


def test_build_bm25_aligns_chunk_ids() -> None:
    chunks = _corpus()
    bm25, ids = build_bm25(chunks)
    assert ids == ["a", "b", "c", "d", "e"]
    scores = bm25.get_scores(tokenize("PEEP"))
    # Only chunk 'a' mentions PEEP.
    assert scores[0] > 0
    assert all(scores[i] == 0 for i in range(1, 5))


def test_save_round_trip(settings: Settings) -> None:
    """save_bm25 + BM25Retriever loads the pickle and resolves chunks via Chroma."""
    chunks = _corpus()
    bm25, ids = build_bm25(chunks)
    save_bm25(settings.bm25_path, bm25, ids)

    # Fake Chroma collection that returns the matching chunk record.
    fake_collection = MagicMock()
    fake_collection.get.return_value = {
        "ids": ["a"],
        "documents": [chunks[0].text],
        "metadatas": [
            {
                "source_path": "/c/a.pdf",
                "source_name": "a",
                "page_number": 1,
                "section_heading": "",
                "token_count": chunks[0].token_count,
            }
        ],
    }

    def fake_get_collection(_settings: Settings) -> MagicMock:
        return fake_collection

    # bm25.py imports get_collection at the top; patch its bound reference.
    import guideline_gpt.retrieval.bm25 as mod

    monkeypatched = mod.get_collection
    mod.get_collection = fake_get_collection  # type: ignore[assignment]
    try:
        retriever = BM25Retriever(settings)
    finally:
        mod.get_collection = monkeypatched  # type: ignore[assignment]

    hits = retriever.search("PEEP", top_k=2)
    assert len(hits) == 1  # 'b' has score 0, omitted
    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].rank == 1
    assert hits[0].source == "bm25"
    assert hits[0].score > 0


def test_missing_index_raises(settings: Settings) -> None:
    settings.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    # File deliberately absent.
    with pytest.raises(FileNotFoundError, match="BM25 index not found"):
        BM25Retriever(settings)


def test_bm25_module_doc() -> None:
    assert bm25_module.__doc__ is not None
