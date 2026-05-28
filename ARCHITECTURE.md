# Architecture

> This document describes the system design of guideline-gpt. It is expanded as
> the implementation lands (milestones M0–M7). For the rationale behind specific
> choices, see the [Architecture Decision Records](docs/decisions/).

## Overview

guideline-gpt is a Retrieval-Augmented Generation system split into an **offline
ingestion** path and an **online query** path. A single `QueryTrace` object
threads through the query path and is the contract between the pipeline and the
UI.

```
┌──────────────────────────────────────────────────────────────┐
│                       Ingestion (offline)                      │
│   PDFs ──► Parser ──► Chunker ──► Embedder ──► ChromaDB        │
│                          └──────────────────► BM25 index       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       Query (online)                           │
│   query ─┬─► Vector search ─┐                                  │
│          └─► BM25 search ───┴─► RRF fusion ─► rerank ─► top-k   │
│                          ─► prompt assembly ─► LLM ─► answer    │
└──────────────────────────────────────────────────────────────┘
```

## Module map

| Package | Responsibility |
|---|---|
| `config` | Pydantic `Settings`; all tunables live here. |
| `types` | Shared dataclasses (`Chunk`, `RetrievalHit`, `QueryTrace`, …). |
| `logging_setup` | structlog configuration. |
| `ingestion/` | PDF loading, chunking, index building. |
| `retrieval/` | Vector, BM25, RRF fusion, cross-encoder rerank. |
| `generation/` | Provider-agnostic LLM client, prompts, answer assembly. |
| `pipeline` | End-to-end query orchestration + trace construction. |
| `ui/` | Streamlit inspector (no business logic). |

## Key principles

1. **The trace is sacred.** Every stage appends to `QueryTrace`; failures are
   recorded, not swallowed.
2. **No untyped dicts cross module boundaries.** Use the dataclasses in `types`.
3. **Configuration is centralized** in `Settings`.
4. **The UI is a thin layer** — it calls `pipeline.query()` and renders.
5. **Provider parity** — anything that works with one LLM provider works with the
   other.

## Data flow contract

The `QueryTrace` dataclass (see `types.py`) records, per query: the raw query,
each retrieval stage's hits, the assembled prompt, LLM metadata (provider,
model, latency, token counts), the final answer, parsed citations, and any
per-stage errors. The inspector UI renders exactly these fields.
