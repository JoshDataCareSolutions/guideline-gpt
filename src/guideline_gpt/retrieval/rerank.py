"""Cross-encoder reranking of retrieval candidates.

A cross-encoder scores (query, chunk) pairs jointly — unlike a bi-encoder that
embeds them independently — so its relevance signal is much stronger but more
expensive. We rerank only the small candidate set surfaced by hybrid retrieval.

The default model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) runs comfortably
on CPU. It is downloaded and cached on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guideline_gpt.config import Settings
from guideline_gpt.logging_setup import get_logger
from guideline_gpt.types import RetrievalHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = get_logger(__name__)


class CrossEncoderReranker:
    """Reorders retrieval hits by cross-encoder relevance scores."""

    def __init__(self, settings: Settings, *, model: CrossEncoder | None = None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading_cross_encoder", model=settings.rerank_model)
            model = CrossEncoder(settings.rerank_model)
        self._model = model

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        """Rescore ``hits`` jointly against ``query`` and keep the top ``top_k``.

        Args:
            query: The natural-language query.
            hits: Candidate hits to rerank (typically the fused output).
            top_k: Number of hits to keep.

        Returns:
            The top hits in descending relevance order, with ``source="rerank"``
            and fresh 1-indexed ranks.
        """
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievalHit(chunk=hit.chunk, score=float(score), source="rerank", rank=rank)
            for rank, (hit, score) in enumerate(ranked[:top_k], start=1)
        ]
