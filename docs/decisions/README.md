# Architecture Decision Records

This directory records the significant architectural decisions made on
guideline-gpt, using lightweight [ADRs](https://adr.github.io/). Each record
captures the context, the decision, and its consequences so the *why* behind a
choice survives even when the code changes.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-hybrid-retrieval-with-rrf-fusion.md) | Hybrid retrieval (vector + BM25) with RRF fusion | Accepted |
| [0003](0003-chromadb-as-vector-store.md) | ChromaDB as the vector store | Accepted |
| [0004](0004-provider-agnostic-llm-client.md) | Provider-agnostic LLM client | Accepted |
| [0005](0005-api-free-cross-encoder-reranker.md) | API-free local cross-encoder reranker | Accepted |

New ADRs are numbered sequentially and never deleted; a superseded decision is
marked as such and points to the record that replaces it.
