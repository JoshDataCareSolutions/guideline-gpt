# 3. ChromaDB as the vector store

- **Status:** Accepted
- **Date:** 2026-01-22

## Context

The system needs a vector store for dense retrieval. For an educational,
single-node showcase that anyone can run in ~5 minutes (or one `docker compose
up`), the priorities are: zero external services to provision, on-disk
persistence, a built-in embedding-function abstraction, and a small, readable
API. Hosted options (Pinecone, Azure AI Search) and server-based stores
(pgvector, Weaviate, Qdrant) all add operational setup that works against the
"clone and run" goal.

## Decision

We use **ChromaDB** in `PersistentClient` mode, with the collection created using
an OpenAI embedding function so that ingestion and querying always embed through
the same configured model. Distance is configured as **cosine** so scores
normalize cleanly to `[0, 1]` via `1 - distance / 2`. See `retrieval/vector.py`.

Because Chroma `upsert` adds and updates but never removes, ingestion explicitly
**reconciles membership**: before upserting the current chunk set it deletes any
stored id that is no longer present, so the vector store stays consistent with
the freshly rebuilt BM25 index instead of accumulating orphans from deleted or
edited documents. See `ingestion/pipeline.py`.

## Consequences

- No external service to stand up; the store is a directory on disk
  (`chroma_db/`), trivial to delete and rebuild.
- The same embedding function is reused for documents and queries, removing a
  common source of train/serve skew.
- Re-ingestion is idempotent *and* drift-free: removing a PDF and re-ingesting
  prunes its chunks from both stores.
- Chroma is single-node; horizontal scale and high-availability are out of
  scope for this showcase. Swapping in a server-backed store later is isolated
  behind `VectorRetriever` and would be its own ADR (listed in the roadmap).
