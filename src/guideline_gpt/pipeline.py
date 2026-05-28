"""End-to-end query pipeline that builds a complete :class:`QueryTrace`.

The trace is the contract with the UI: every stage appends to it, and a stage
failure is recorded in ``trace.stage_errors`` rather than aborting the query.

At M2 the retrieval stage is vector-only; hybrid fusion (M3) and cross-encoder
reranking (M4) slot into :meth:`QueryPipeline._retrieve` as they land.
"""

from __future__ import annotations

from datetime import datetime

from guideline_gpt.config import Settings
from guideline_gpt.generation.answer import parse_citations
from guideline_gpt.generation.llm_client import LLMClient, get_llm_client
from guideline_gpt.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from guideline_gpt.logging_setup import get_logger
from guideline_gpt.retrieval.bm25 import BM25Retriever
from guideline_gpt.retrieval.hybrid import fuse
from guideline_gpt.retrieval.rerank import CrossEncoderReranker
from guideline_gpt.retrieval.vector import VectorRetriever
from guideline_gpt.types import Chunk, QueryResponse, QueryTrace, RetrievalHit

log = get_logger(__name__)

# Per-million-token prices (USD), matched by model-name prefix. Keep cheap models
# as the default so cost stays low; update as provider pricing changes.
_PRICES: dict[str, tuple[float, float]] = {
    # model prefix: (input_per_1M, output_per_1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a completion from token counts.

    Args:
        model: The model name (matched by known prefix).
        input_tokens: Prompt tokens.
        output_tokens: Generated tokens.

    Returns:
        Estimated cost in USD; ``0.0`` if the model's pricing is unknown.
    """
    for prefix, (in_price, out_price) in _PRICES.items():
        if model.startswith(prefix):
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return 0.0


class QueryPipeline:
    """Orchestrates retrieval and generation, producing a cited answer + trace."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm_client: LLMClient | None = None,
        vector: VectorRetriever | None = None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._settings = settings
        self._vector = vector or VectorRetriever(settings)
        self._bm25 = bm25 or BM25Retriever(settings)
        self._reranker = reranker or CrossEncoderReranker(settings)
        self._llm = llm_client or get_llm_client(settings)

    def _retrieve(self, query: str, trace: QueryTrace) -> list[RetrievalHit]:
        """Run retrieval and return the hits whose chunks are sent to the LLM.

        Pipeline: vector + BM25 -> RRF fusion -> cross-encoder rerank. Each
        stage's output is recorded on the trace; a stage failure is logged but
        does not abort the query (see "the trace is sacred").
        """
        retrieval_k = self._settings.retrieval_top_k
        rerank_k = self._settings.rerank_top_k
        vector_hits: list[RetrievalHit] = []
        bm25_hits: list[RetrievalHit] = []

        try:
            vector_hits = self._vector.search(query, retrieval_k)
            trace.vector_hits = vector_hits
        except Exception as exc:  # noqa: BLE001 - record and continue per trace contract
            log.warning("vector_search_failed", error=str(exc))
            trace.stage_errors["vector"] = str(exc)

        try:
            bm25_hits = self._bm25.search(query, retrieval_k)
            trace.bm25_hits = bm25_hits
        except Exception as exc:  # noqa: BLE001 - record and continue per trace contract
            log.warning("bm25_search_failed", error=str(exc))
            trace.stage_errors["bm25"] = str(exc)

        try:
            fused = fuse(vector_hits, bm25_hits, retrieval_k)
            trace.fused_hits = fused
        except Exception as exc:  # noqa: BLE001 - record and continue per trace contract
            log.warning("fusion_failed", error=str(exc))
            trace.stage_errors["fusion"] = str(exc)
            fused = vector_hits or bm25_hits

        try:
            reranked = self._reranker.rerank(query, fused, rerank_k)
            trace.reranked_hits = reranked
        except Exception as exc:  # noqa: BLE001 - record and continue per trace contract
            log.warning("rerank_failed", error=str(exc))
            trace.stage_errors["rerank"] = str(exc)
            # Fall back to the top fused hits so the LLM still gets context.
            return fused[:rerank_k]

        return reranked

    def _generate(self, query: str, chunks: list[Chunk], trace: QueryTrace) -> None:
        """Assemble the prompt, call the LLM, and populate the trace."""
        user_prompt = build_user_prompt(query, chunks)
        trace.prompt_assembled = f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== USER ===\n{user_prompt}"
        trace.llm_provider = self._settings.llm_provider
        try:
            result = self._llm.complete(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001 - record and continue per trace contract
            log.warning("generation_failed", error=str(exc))
            trace.stage_errors["generation"] = str(exc)
            return
        trace.answer = result.text
        trace.llm_model = result.model
        trace.llm_latency_ms = result.latency_ms
        trace.llm_input_tokens = result.input_tokens
        trace.llm_output_tokens = result.output_tokens
        trace.citations = parse_citations(result.text)

    @staticmethod
    def _cited_chunks(chunks: list[Chunk], citations: list[str]) -> list[Chunk]:
        """Map citation markers (1-indexed) back to their context chunks."""
        cited: list[Chunk] = []
        for marker in citations:
            idx = int(marker) - 1
            if 0 <= idx < len(chunks):
                cited.append(chunks[idx])
        return cited or chunks

    def query(self, question: str) -> QueryResponse:
        """Answer ``question`` end-to-end, returning the answer, sources, and trace."""
        trace = QueryTrace(query=question, timestamp=datetime.now())
        log.info("query_start", query=question)

        context_hits = self._retrieve(question, trace)
        context_chunks = [hit.chunk for hit in context_hits]

        self._generate(question, context_chunks, trace)
        citations = self._cited_chunks(context_chunks, trace.citations)

        log.info(
            "query_done",
            answered=bool(trace.answer),
            context_chunks=len(context_chunks),
            citations=len(citations),
            errors=list(trace.stage_errors),
        )
        return QueryResponse(answer=trace.answer, citations=citations, trace=trace)
