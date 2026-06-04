# 2. Hybrid retrieval (vector + BM25) with RRF fusion

- **Status:** Accepted
- **Date:** 2026-01-20

## Context

Dense (embedding) retrieval generalizes well over paraphrase and synonymy but
can miss exact terms — drug names, dosages, abbreviations, and section codes that
appear verbatim in clinical guidelines. Lexical retrieval (BM25) is the
complement: excellent on exact tokens, blind to paraphrase. A system that relies
on only one of these has a predictable class of misses.

Combining the two raises a calibration problem: cosine similarity scores and
BM25 scores live on different, query-dependent scales, so we cannot simply add
or average them.

## Decision

We run **both** retrievers and fuse their results with **Reciprocal Rank Fusion**
(Cormack et al., 2009). RRF combines ranked lists using only the *rank* of each
item, not its score:

```
score(c) = sum over each list containing c of  1 / (k + rank_in_list)
```

This sidesteps score-scale calibration entirely. We use **k = 60**, the value
from the original paper and the de-facto default in widely used RAG stacks
(e.g. Microsoft's implementations). See `retrieval/hybrid.py`.

**BM25 tokenization** is deliberately minimal: lowercase, then keep alphanumeric
runs (punctuation dropped). We do **not** stem. Clinical text is dense with
tokens where stemming hurts — "GOLD" vs "gold", drug suffixes, and numeric
dose/unit tokens — and the cross-encoder reranker (ADR-005) recovers most
morphological matching downstream. Keeping tokenization simple also keeps the
lexical layer transparent in the pipeline inspector. See `retrieval/bm25.py`.

The BM25 index stores only `(BM25Okapi, chunk_ids)`; chunk *content* is resolved
from ChromaDB by id at query time so the lexical and dense stores cannot drift
in content. (Membership is kept in sync at ingest — see ADR-003.)

## Consequences

- Recall improves on both paraphrased and exact-term queries versus either
  retriever alone.
- No score normalization to maintain or get wrong.
- `k = 60` is a tunable constant, not sacred; it is centralized as `RRF_K`.
- Two indexes must be built at ingest and kept membership-consistent.
- Skipping stemming means a handful of purely morphological lexical matches are
  left to the reranker rather than the BM25 stage.
