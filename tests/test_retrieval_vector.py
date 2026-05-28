"""Tests for vector record (de)serialization and search scoring (mocked Chroma)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from guideline_gpt import retrieval
from guideline_gpt.config import Settings
from guideline_gpt.retrieval import vector as vector_module
from guideline_gpt.retrieval.vector import (
    VectorRetriever,
    chunk_to_record,
    record_to_chunk,
)
from guideline_gpt.types import Chunk, DocumentMetadata


def _chunk(chunk_id: str = "abc123", heading: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="Maintain PEEP at recommended levels.",
        metadata=DocumentMetadata(
            source_path="/corpus/ards.pdf",
            source_name="ards",
            page_number=12,
            section_heading=heading,
        ),
        token_count=7,
    )


def test_record_round_trip_no_heading() -> None:
    original = _chunk()
    cid, doc, meta = chunk_to_record(original)
    assert cid == original.chunk_id
    assert doc == original.text
    restored = record_to_chunk(cid, doc, meta)
    assert restored == original


def test_record_round_trip_with_heading() -> None:
    original = _chunk(heading="Ventilation")
    cid, doc, meta = chunk_to_record(original)
    assert meta["section_heading"] == "Ventilation"
    assert record_to_chunk(cid, doc, meta) == original


def test_search_normalizes_scores_and_ranks(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "ids": [["a", "b"]],
        "documents": [["text a", "text b"]],
        "metadatas": [
            [
                {
                    "source_path": "/c/a.pdf",
                    "source_name": "a",
                    "page_number": 1,
                    "section_heading": "",
                    "token_count": 3,
                },
                {
                    "source_path": "/c/b.pdf",
                    "source_name": "b",
                    "page_number": 2,
                    "section_heading": "",
                    "token_count": 4,
                },
            ]
        ],
        "distances": [[0.2, 1.0]],
    }
    monkeypatch.setattr(vector_module, "get_collection", lambda _settings: fake_collection)

    hits = VectorRetriever(settings).search("peep?", top_k=2)

    assert [h.rank for h in hits] == [1, 2]
    assert all(h.source == "vector" for h in hits)
    # cosine distance -> similarity: 1 - d/2
    assert hits[0].score == pytest.approx(0.9)
    assert hits[1].score == pytest.approx(0.5)
    fake_collection.query.assert_called_once_with(query_texts=["peep?"], n_results=2)


def test_retrieval_package_importable() -> None:
    assert retrieval.__doc__ is not None
