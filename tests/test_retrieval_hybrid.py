"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from guideline_gpt.retrieval.hybrid import RRF_K, fuse
from guideline_gpt.types import Chunk, DocumentMetadata, RetrievalHit


def _hit(chunk_id: str, rank: int, source: str = "vector") -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"text {chunk_id}",
            metadata=DocumentMetadata(
                source_path=f"/c/{chunk_id}.pdf",
                source_name=chunk_id,
                page_number=1,
            ),
            token_count=1,
        ),
        score=1.0 / rank,
        source=source,  # type: ignore[arg-type]
        rank=rank,
    )


def test_rrf_score_math() -> None:
    vec = [_hit("a", 1), _hit("b", 2)]
    bm = [_hit("a", 3, "bm25"), _hit("c", 1, "bm25")]
    result = fuse(vec, bm, top_k=10)

    # 'a' appears in both: 1/(60+1) + 1/(60+3)
    # 'c' bm25 only: 1/(60+1)
    # 'b' vector only: 1/(60+2)
    by_id = {hit.chunk.chunk_id: hit for hit in result}
    assert by_id["a"].score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 3))
    assert by_id["c"].score == pytest.approx(1 / (RRF_K + 1))
    assert by_id["b"].score == pytest.approx(1 / (RRF_K + 2))


def test_rrf_dedupes_by_chunk_id_and_sorts_by_score() -> None:
    vec = [_hit("a", 1), _hit("b", 2)]
    bm = [_hit("a", 3, "bm25")]
    result = fuse(vec, bm, top_k=10)
    ids = [hit.chunk.chunk_id for hit in result]
    assert ids.count("a") == 1
    # 'a' beats 'b' (it's in both lists, with strong vector rank).
    assert ids[0] == "a"
    assert all(hit.source == "hybrid" for hit in result)
    assert [hit.rank for hit in result] == list(range(1, len(result) + 1))


def test_rrf_respects_top_k() -> None:
    vec = [_hit(c, i + 1) for i, c in enumerate(["a", "b", "c", "d"])]
    bm = [_hit(c, i + 1, "bm25") for i, c in enumerate(["e", "f", "g"])]
    result = fuse(vec, bm, top_k=3)
    assert len(result) == 3


def test_rrf_handles_empty_lists() -> None:
    assert fuse([], [], top_k=5) == []
    only_vector = fuse([_hit("a", 1)], [], top_k=5)
    assert len(only_vector) == 1 and only_vector[0].chunk.chunk_id == "a"
    only_bm = fuse([], [_hit("z", 1, "bm25")], top_k=5)
    assert len(only_bm) == 1 and only_bm[0].chunk.chunk_id == "z"
