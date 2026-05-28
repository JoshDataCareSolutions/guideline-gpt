"""Tests for cross-encoder reranking (mocked model — no torch needed)."""

from __future__ import annotations

from unittest.mock import MagicMock

from guideline_gpt.config import Settings
from guideline_gpt.retrieval.rerank import CrossEncoderReranker
from guideline_gpt.types import Chunk, DocumentMetadata, RetrievalHit


def _hit(chunk_id: str, text: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=DocumentMetadata(
                source_path=f"/c/{chunk_id}.pdf",
                source_name=chunk_id,
                page_number=1,
            ),
            token_count=len(text.split()),
        ),
        score=0.5,
        source="hybrid",
        rank=rank,
    )


def test_rerank_reorders_by_predicted_scores(settings: Settings) -> None:
    fake_model = MagicMock()
    # Hits arrive in input order [a, b, c]; predicted scores invert that.
    fake_model.predict.return_value = [0.1, 0.9, 0.4]

    hits = [_hit("a", "ARDS", 1), _hit("b", "PEEP", 2), _hit("c", "COPD", 3)]
    out = CrossEncoderReranker(settings, model=fake_model).rerank("PEEP?", hits, top_k=3)

    assert [h.chunk.chunk_id for h in out] == ["b", "c", "a"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert all(h.source == "rerank" for h in out)
    assert out[0].score == 0.9
    # The (query, text) pairs were passed in input order, not reordered.
    fake_model.predict.assert_called_once_with(
        [("PEEP?", "ARDS"), ("PEEP?", "PEEP"), ("PEEP?", "COPD")]
    )


def test_rerank_respects_top_k(settings: Settings) -> None:
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9, 0.4, 0.6]
    hits = [_hit(c, c, i + 1) for i, c in enumerate(["a", "b", "c", "d"])]
    out = CrossEncoderReranker(settings, model=fake_model).rerank("q", hits, top_k=2)
    assert [h.chunk.chunk_id for h in out] == ["b", "d"]


def test_rerank_empty_input(settings: Settings) -> None:
    fake_model = MagicMock()
    assert CrossEncoderReranker(settings, model=fake_model).rerank("q", [], top_k=5) == []
    fake_model.predict.assert_not_called()
