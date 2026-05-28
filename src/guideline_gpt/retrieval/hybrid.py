"""Reciprocal Rank Fusion of dense and lexical retrieval results.

RRF (Cormack et al., 2009) combines results from multiple retrievers using only
their ranks, sidestepping the need to calibrate dense and BM25 score scales. For
each candidate ``c`` appearing in any list, its fused score is::

    score(c) = sum over each list containing c of 1 / (k + rank_in_list)

with ``k = 60`` (the de-facto default in the literature and Microsoft's RAG
implementations; see ADR-002).
"""

from __future__ import annotations

from guideline_gpt.types import Chunk, RetrievalHit

# RRF smoothing constant. 60 is the canonical value from the original paper.
RRF_K = 60


def fuse(
    vector_hits: list[RetrievalHit],
    bm25_hits: list[RetrievalHit],
    top_k: int,
) -> list[RetrievalHit]:
    """Combine two ranked retrieval lists via Reciprocal Rank Fusion.

    Chunks present in both lists have their contributions summed. The chunk
    object itself is taken from the first list that surfaced it (vector first,
    then BM25) — they refer to the same persisted chunk by id.

    Args:
        vector_hits: Dense retrieval hits, ranked.
        bm25_hits: Lexical retrieval hits, ranked.
        top_k: Maximum number of fused hits to return.

    Returns:
        Fused hits with ``source="hybrid"`` and fresh 1-indexed ranks. The
        ``score`` field holds the RRF score.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    # Vector list first so a chunk seen in both keeps the vector reference.
    for source_list in (vector_hits, bm25_hits):
        for hit in source_list:
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + hit.rank)
            chunks.setdefault(cid, hit.chunk)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        RetrievalHit(chunk=chunks[cid], score=score, source="hybrid", rank=rank)
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]
