# 5. API-free local cross-encoder reranker

- **Status:** Accepted
- **Date:** 2026-01-28

## Context

Hybrid retrieval (ADR-002) returns a candidate set that is good on recall but
only roughly ordered: RRF ranks by fused position, not by a direct judgment of
how well a chunk answers *this* query. A reranking stage that scores each
`(query, chunk)` pair jointly sharply improves precision at the top, which is
what matters because only the top few chunks are sent to the LLM.

The options are a hosted reranking API (e.g. Cohere Rerank) or a local
cross-encoder model. A hosted API means another key, another vendor dependency,
per-call cost, network latency, and sending query text off-box.

## Decision

We rerank with a **local cross-encoder**, defaulting to
`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`. A
cross-encoder embeds the query and chunk *together* and outputs a single
relevance score, a much stronger signal than the bi-encoder used for first-stage
dense retrieval — at a cost that is acceptable because we only rerank the small
hybrid candidate set, not the whole corpus. The model is small enough to run
comfortably on CPU and is downloaded and cached on first use, so the system has
**no reranking API dependency, no extra key, and no per-call cost**. See
`retrieval/rerank.py`.

The model is configurable (`RERANK_MODEL`), and the reranker is constructed with
dependency injection so tests can supply a stub instead of loading weights.

## Consequences

- Strong precision gains at the top of the ranking with no external service.
- Reranking runs offline and keeps query text on-box.
- First run pays a one-time model download; the Docker image pre-fetches it so
  container startup does not block on it.
- Reranking adds CPU latency per query (bounded, since only the candidate set is
  scored). A hosted reranker (e.g. Cohere) remains a future option and is listed
  in the roadmap; swapping it in is isolated behind `CrossEncoderReranker`.
