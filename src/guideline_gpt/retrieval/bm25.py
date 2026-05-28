"""BM25 lexical retrieval, persisted alongside the vector store.

Tokenization is deliberately simple: lowercase, then extract alphanumeric runs
(dropping punctuation). This choice — and why we don't stem — is recorded in
ADR-002. The index is pickled as ``(BM25Okapi, chunk_ids)``; chunk content is
resolved from the Chroma collection at query time so the two stores never drift.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from guideline_gpt.config import Settings
from guideline_gpt.logging_setup import get_logger
from guideline_gpt.retrieval.vector import get_collection, record_to_chunk
from guideline_gpt.types import Chunk, RetrievalHit

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, then alphanumeric runs only."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25 index over the chunk texts.

    Args:
        chunks: The corpus chunks, in a stable order.

    Returns:
        A tuple of the fitted BM25 index and the aligned list of chunk ids.
    """
    corpus = [tokenize(chunk.text) for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    return BM25Okapi(corpus), chunk_ids


def save_bm25(path: Path, bm25: BM25Okapi, chunk_ids: list[str]) -> None:
    """Persist the BM25 index and aligned chunk ids to ``path`` via pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, fh)
    log.info("bm25_saved", path=str(path), chunks=len(chunk_ids))


class BM25Retriever:
    """Lexical retrieval backed by a pickled BM25 index.

    Chunk content is resolved from the Chroma collection by id, keeping the
    lexical and dense stores consistent.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {settings.bm25_path}. Run `guideline-gpt ingest` first."
            )
        with settings.bm25_path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - index is produced by our own pipeline
        self._bm25: BM25Okapi = payload["bm25"]
        self._chunk_ids: list[str] = payload["chunk_ids"]
        self._collection = get_collection(settings)

    def _resolve_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Fetch chunk content for the given ids from the vector store."""
        if not chunk_ids:
            return {}
        result = self._collection.get(ids=chunk_ids)
        resolved: dict[str, Chunk] = {}
        for cid, doc, meta in zip(
            result["ids"],
            result["documents"] or [],
            result["metadatas"] or [],
            strict=True,
        ):
            resolved[cid] = record_to_chunk(cid, doc, meta)
        return resolved

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return the top-k chunks for ``query`` by BM25 score.

        Args:
            query: The natural-language query.
            top_k: Number of hits to return.

        Returns:
            Ranked retrieval hits (rank is 1-indexed). Chunks with a score of
            zero are omitted.
        """
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        top = [(self._chunk_ids[idx], float(score)) for idx, score in ranked if score > 0]

        resolved = self._resolve_chunks([cid for cid, _ in top])
        hits: list[RetrievalHit] = []
        for rank, (cid, score) in enumerate(top, start=1):
            chunk = resolved.get(cid)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=score, source="bm25", rank=rank))
        return hits
