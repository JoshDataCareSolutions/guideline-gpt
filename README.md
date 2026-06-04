# guideline-gpt

[![CI](https://github.com/joshquigley/guideline-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/joshquigley/guideline-gpt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://docs.astral.sh/ruff/)

A production-pattern **Retrieval-Augmented Generation (RAG)** system for
querying domain-specific document corpora — built around a **pipeline inspector
UI** that exposes *how the system works* at every stage, not just a chat box.

The default showcase corpus is public clinical guidelines (respiratory and
critical care), but the system is **corpus-agnostic**: drop your own PDFs into
`documents/`, rebuild the index, and query your own domain.

> **Not a medical device.** This project is for education and engineering
> demonstration only — not for clinical decision-making. It processes only
> public documents and performs no PHI handling. See the [disclaimer](#disclaimer).

---

## Table of contents

- [What is RAG, and why does it matter?](#what-is-rag-and-why-does-it-matter)
- [What makes this implementation different](#what-makes-this-implementation-different)
- [The retrieval pipeline, stage by stage](#the-retrieval-pipeline-stage-by-stage)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quickstart](#quickstart-local-5-minutes)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Engineering principles](#engineering-principles)
- [Testing & CI](#testing--ci)
- [Project layout](#project-layout)
- [Further reading](#further-reading)

---

## What is RAG, and why does it matter?

Large language models are trained on a fixed snapshot of the internet. That
creates three problems for any serious application:

1. **Knowledge cutoff** — the model doesn't know anything published after
   training.
2. **No private/domain knowledge** — your internal documents, a specific set of
   clinical guidelines, a product manual: the model has never seen them.
3. **Hallucination** — when a model doesn't know something, it tends to invent a
   confident, plausible-sounding answer rather than admit uncertainty.

**Retrieval-Augmented Generation** solves all three by *separating knowledge from
reasoning*. Instead of relying on the model's frozen parameters, RAG:

1. **Retrieves** the most relevant passages from a trusted document corpus, then
2. **Augments** the model's prompt with those passages as context, so the model
3. **Generates** an answer grounded in — and citing — that retrieved evidence.

```
                         ┌─────────────┐
   Question  ───────────▶│  RETRIEVER  │──────▶  top-k relevant passages
                         └─────────────┘                  │
                                                          ▼
                         ┌─────────────┐         "Answer using ONLY these
   Grounded answer ◀─────│  GENERATOR  │◀──────   excerpts: [1]… [2]… [3]…"
   with [1][2] citations └─────────────┘         (passages + question + rules)
```

The model becomes a *reasoning engine over supplied evidence* rather than an
oracle drawing on half-remembered training data. The answer is traceable to a
source, the corpus can be updated without retraining, and the model is
explicitly instructed to refuse when the evidence is insufficient.

The hard part of RAG is not the "G" — calling an LLM is one API request. **The
hard part is the "R".** If retrieval surfaces the wrong passages, even the best
model produces a confidently wrong answer. This project treats retrieval as a
first-class, multi-stage problem and makes every stage observable.

---

## What makes this implementation different

Most RAG demos are a single chat box that hides everything interesting behind it.
This one is built around a **pipeline inspector**: a two-pane Streamlit UI where
the left side is the chat and the right side shows, for the *current* query:

- the **raw candidates** from each retriever (vector and keyword), side by side;
- the **fused ranking** after Reciprocal Rank Fusion;
- the **reranked top-k** with cross-encoder relevance scores — the exact chunks
  sent to the model;
- the **fully assembled prompt** (system + user) the LLM actually received;
- **LLM metadata** — provider, model, input/output token counts, latency, and an
  estimated per-query **cost in USD**.

Everything the inspector renders comes from a single `QueryTrace` object that the
pipeline populates as it runs. **The trace is the contract** between the engine
and the UI — the UI contains no business logic; it is a pure rendering of the
trace. This makes the system easy to reason about, easy to test, and genuinely
educational: you can *watch* retrieval succeed or fail.

---

## The retrieval pipeline, stage by stage

This is where the engineering lives. Each stage exists to fix a specific failure
mode of the stage before it.

### 1. Ingestion — turning PDFs into a searchable index *(offline)*

```
PDFs ─▶ extract text ─▶ strip boilerplate ─▶ chunk ─▶ embed ─▶ ChromaDB
                                                 └──────────▶ BM25 index
```

- **Extraction** uses `pypdf` with a `pymupdf` fallback for awkward/encoded PDFs.
- **Boilerplate stripping** detects lines that repeat on ≥50% of a document's
  pages — running headers, footers, copyright notices — and removes them before
  chunking, so they don't pollute embeddings or dilute relevance.
- **Chunking** splits text into ~512-token passages with 64-token overlap using a
  sentence-aware splitter and the `cl100k_base` tokenizer (the same one OpenAI
  embeddings use, so token budgets line up). Overlap prevents a relevant sentence
  from being orphaned at a chunk boundary.
- **Stable chunk IDs** are a SHA-256 hash of `source · page · text`. This makes
  re-ingestion **idempotent**: unchanged content yields the same ID and is
  upserted in place, deleted PDFs have their orphaned chunks pruned, and the
  vector store and keyword index never drift out of sync.

> **Why two indexes?** See the next two stages — dense and sparse retrieval fail
> in opposite ways, so we build and query both.

### 2. Dense retrieval (vector search) — *meaning*

Each chunk is embedded into a high-dimensional vector with OpenAI
`text-embedding-3-small` and stored in **ChromaDB**. At query time the question
is embedded the same way, and we retrieve the chunks whose vectors are closest by
**cosine similarity**.

Dense retrieval captures **semantic** similarity: a query for *"shortness of
breath"* can match a passage about *"dyspnea"* even with zero shared words.

**Its weakness:** it can miss exact terms — a specific drug name, dosage, acronym,
or error code — because rare tokens get smoothed into the embedding.

### 3. Sparse retrieval (BM25) — *exact terms*

In parallel, the same query runs through **BM25** (the Okapi ranking function),
the classic keyword-relevance algorithm behind decades of search engines. BM25
scores chunks on term frequency and rarity — it excels at exactly what dense
retrieval misses: precise lexical matches like `FEV1`, `SpO2`, or a specific
guideline grade.

**Its weakness:** it's blind to meaning — no shared words, no match.

### 4. Hybrid fusion (Reciprocal Rank Fusion) — *best of both*

Dense and sparse retrievers each return a ranked list, but their scores live on
incompatible scales (cosine distance vs. BM25 magnitude). Naively averaging them
is meaningless. **Reciprocal Rank Fusion (RRF)** sidesteps calibration entirely
by combining *ranks*, not scores:

```
score(chunk) = Σ  1 / (k + rank_in_list)        with k = 60
            over each list the chunk appears in
```

A chunk ranked highly by *both* retrievers floats to the top; a chunk that only
one retriever found still gets credit. `k = 60` is the value from the original
RRF paper and a common production default — large enough that top ranks aren't
wildly dominant, small enough that rank still matters. The rationale is recorded
in [ADR-0002](docs/decisions/0002-hybrid-retrieval-with-rrf-fusion.md).

### 5. Reranking (cross-encoder) — *precision*

Fusion gives us ~20 decent candidates, but it's still based on the *initial*
retrieval signals. The final stage applies a **cross-encoder** reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reads the **query and each chunk
together** and scores their true relevance.

This is more accurate than embedding similarity because the model attends to the
query and the passage jointly, rather than comparing two independently-computed
vectors. It's also more expensive — which is exactly why it runs *last*, on a
small candidate set, rather than over the whole corpus. The two-phase
**retrieve-then-rerank** pattern (cheap-and-broad → expensive-and-precise) is the
standard way to get both recall and precision.

The reranker runs locally on CPU via `sentence-transformers` — no API cost, and
the query text never leaves the box. See
[ADR-0005](docs/decisions/0005-api-free-cross-encoder-reranker.md).

The top **5** reranked chunks are what the LLM actually sees.

### 6. Generation — *grounded, cited, and willing to say "no"*

The selected chunks are formatted into a numbered, source-attributed context
block and combined with a system prompt that enforces three rules:

> - Answer **strictly** from the numbered excerpts. Do not use outside knowledge.
> - **Cite every claim** with `[n]` markers pointing at the excerpt that supports
>   it.
> - If the excerpts don't cover the question, **say so plainly** and don't guess.

That refusal clause is the antidote to hallucination — the most important and most
overlooked part of a trustworthy RAG system. After generation, citation markers
are parsed out of the answer and mapped back to their source chunks so the UI can
show exactly which passage backs each claim.

The LLM is **provider-agnostic** behind a single `LLMClient` protocol: switch
between Anthropic (`claude-haiku-4-5`, the default) and OpenAI (`gpt-4o-mini`)
with one environment variable, with identical behavior either way
([ADR-0004](docs/decisions/0004-provider-agnostic-llm-client.md)).

---

## Architecture

guideline-gpt is split into an **offline ingestion** path and an **online query**
path. A single `QueryTrace` object threads through the query path and is the
contract between the pipeline and the UI.

```
┌────────────────────────────────────────────────────────────────────┐
│                         INGESTION (offline)                          │
│                                                                      │
│   PDFs ──▶ Loader ──▶ Chunker ──▶ Embedder ──▶ ChromaDB (vectors)    │
│           (pypdf/    (512-tok,                                        │
│            pymupdf,   stable IDs)  └────────▶ BM25 index (keywords)   │
│            de-boiler)                                                 │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                          QUERY (online)                              │
│                                                                      │
│   query ─┬─▶ Vector search ─┐                                        │
│          └─▶ BM25 search ───┴─▶ RRF fusion ─▶ Cross-encoder rerank   │
│                                                        │             │
│                          ┌─────────────────────────────┘             │
│                          ▼                                            │
│            Prompt assembly ─▶ LLM ─▶ cited answer + QueryTrace        │
│                                       (rendered by the inspector UI)  │
└────────────────────────────────────────────────────────────────────┘
```

### Module map

| Package | Responsibility |
|---|---|
| `config` | Pydantic `Settings`; every tunable lives here, type-safe and centralized. |
| `types` | Shared dataclasses (`Chunk`, `RetrievalHit`, `QueryTrace`, `QueryResponse`, …). |
| `logging_setup` | Structured JSON logging via `structlog`. |
| `ingestion/` | PDF loading, boilerplate stripping, chunking, index building. |
| `retrieval/` | Vector search, BM25, RRF fusion, cross-encoder reranking. |
| `generation/` | Provider-agnostic LLM client, prompt templates, citation parsing. |
| `pipeline` | End-to-end query orchestration + cost estimation + trace construction. |
| `ui/` | Streamlit inspector — a thin rendering layer with no business logic. |

For the full design narrative see [ARCHITECTURE.md](ARCHITECTURE.md); for the
*why* behind individual choices see the
[Architecture Decision Records](docs/decisions/).

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Mature ML/NLP ecosystem. |
| Vector store | **ChromaDB** (local, persistent) | Zero-config, no external service; cosine distance ([ADR-0003](docs/decisions/0003-chromadb-as-vector-store.md)). |
| Embeddings | OpenAI `text-embedding-3-small` | Strong quality-per-dollar; 1536-dim. |
| Keyword search | `rank-bm25` (BM25Okapi) | Battle-tested lexical baseline for hybrid retrieval. |
| Reranker | `sentence-transformers` cross-encoder (MiniLM) | Local, CPU-friendly, no API cost. |
| LLM | Anthropic **or** OpenAI (pluggable) | Provider parity behind one protocol. |
| Ingestion | `llama-index-core` splitter + `tiktoken` | Sentence-aware, token-accurate chunking. |
| PDF parsing | `pypdf` + `pymupdf` fallback | Robust extraction across PDF quirks. |
| UI | **Streamlit** | Fast path to a rich, inspectable interface. |
| CLI | **Typer** | Clean, typed command surface. |
| Config | **Pydantic Settings** | Validated, env-driven, no magic constants. |
| Logging | **structlog** (JSON) | Machine-readable, contextual logs. |
| Packaging | `pyproject.toml`, `uv` | Reproducible, modern Python packaging. |
| Quality | `ruff`, `mypy --strict`, `pytest` | Lint, strict typing, tests in CI. |
| Deploy | Docker, Docker Compose, Azure Container Apps (Bicep) | One-command local; IaC for cloud. |

---

## Quickstart (local, ~5 minutes)

**Prerequisites:** Python 3.11+, an `OPENAI_API_KEY` (always required — it powers
embeddings), and optionally an `ANTHROPIC_API_KEY` if you use the default LLM
provider.

```bash
# 1. Install (uv recommended; plain pip works too)
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"

# 2. Configure
cp .env.example .env              # add OPENAI_API_KEY (+ ANTHROPIC_API_KEY)

# 3. Add documents and build the index
python scripts/download_corpus.py # fetches sample clinical guidelines…
                                   # …or just drop your own PDFs into documents/
guideline-gpt ingest              # parse → chunk → embed → build indexes

# 4. Launch the inspector UI
guideline-gpt serve               # opens at http://localhost:8501
```

### CLI

```bash
guideline-gpt ingest [DIR]        # ingest PDFs (defaults to DOCUMENTS_DIR)
guideline-gpt serve [--port N]    # launch the Streamlit inspector
guideline-gpt version             # print the installed version
```

### Docker (one command)

```bash
docker compose up --build         # builds the image and serves on :8501
```

`documents/` and `chroma_db/` are mounted as volumes, so your corpus and index
survive container restarts.

---

## Configuration

All configuration is via environment variables, validated by Pydantic. See
[`.env.example`](.env.example) for the complete list. The most-used variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai`. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | LLM when provider is Anthropic. |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM when provider is OpenAI. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model. |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Chunk size and overlap, in tokens. |
| `RETRIEVAL_TOP_K` | `20` | Candidates retrieved before reranking. |
| `RERANK_TOP_K` | `5` | Chunks sent to the LLM after reranking. |
| `BOILERPLATE_PAGE_FRACTION` | `0.5` | Repeat-fraction threshold for header/footer stripping. |

> `OPENAI_API_KEY` is **always required** — it powers embeddings — regardless of
> which LLM provider you select.

---

## Deployment

- **Local / single host:** `docker compose up --build`. Multi-stage Dockerfile
  (uv builder → slim runtime, CPU-only PyTorch to avoid 2 GB CUDA wheels),
  non-root user, and a `/_stcore/health` health check.
- **Azure Container Apps:** infrastructure-as-code in [`infra/`](infra/) — a Bicep
  template provisions an Azure Container Registry, Log Analytics, Key Vault (for
  API keys), a managed identity, and the Container App. One-shot deploy via
  `bash infra/deploy.sh`. Full walkthrough in
  [docs/deploy-azure.md](docs/deploy-azure.md) (~$5–10/month idle).

---

## Engineering principles

These are enforced throughout the codebase and are what make it a *production
pattern* rather than a notebook demo:

1. **The trace is sacred.** Every pipeline stage appends to `QueryTrace`. A stage
   failure is *recorded* in `trace.stage_errors` and the pipeline degrades
   gracefully (e.g. rerank failure falls back to the fused top-k) — failures are
   never silently swallowed, and a partial answer is always explainable.
2. **No untyped dicts cross module boundaries.** Every public function takes and
   returns the dataclasses in `types.py`. The shape of data is always known.
3. **Configuration is centralized.** No module reads `os.environ` directly or
   hides magic constants — everything tunable is in `Settings`.
4. **The UI is a thin layer.** It calls `pipeline.query()` and renders the trace.
   No retrieval or generation logic leaks into the presentation layer.
5. **Provider parity.** Anything that works with one LLM provider works with the
   other, guaranteed by a single `LLMClient` protocol.
6. **Idempotent ingestion.** Deterministic chunk IDs + orphan pruning keep the
   vector store and BM25 index in lockstep across re-ingestion.
7. **Dependency injection everywhere.** Retrievers, the reranker, and the LLM
   client are all injectable, so each stage is unit-testable in isolation.

---

## Testing & CI

```bash
ruff check src tests          # lint
ruff format src tests         # format
mypy src                      # strict static type-checking
pytest --cov=guideline_gpt    # tests + coverage
```

The test suite (`tests/`) covers the chunker (token counting, ID stability),
PDF loader (extraction, boilerplate detection), ingestion pipeline, all three
retrieval stages (vector, BM25, RRF fusion), the reranker, and both LLM clients —
the business logic, isolated from the UI. Every push runs lint, format-check,
strict `mypy`, and the test suite via
[GitHub Actions](.github/workflows/ci.yml).

---

## Project layout

```
guideline-gpt/
├── src/guideline_gpt/
│   ├── config.py            # Pydantic Settings — all tunables
│   ├── types.py             # Shared dataclasses (Chunk, RetrievalHit, QueryTrace…)
│   ├── pipeline.py          # End-to-end query orchestration + cost estimation
│   ├── cli.py               # Typer CLI (ingest / serve / version)
│   ├── ingestion/           # loader, chunker, ingestion pipeline
│   ├── retrieval/           # vector, bm25, hybrid (RRF), rerank
│   ├── generation/          # llm_client, prompts, answer/citation parsing
│   └── ui/                  # Streamlit inspector + components
├── tests/                   # pytest suite (business logic)
├── scripts/download_corpus.py   # idempotent sample-corpus downloader
├── infra/                   # Azure Bicep + deploy script
├── docs/
│   ├── decisions/           # Architecture Decision Records (ADRs)
│   └── deploy-azure.md
├── ARCHITECTURE.md          # system design narrative
├── Dockerfile / docker-compose.yml
└── pyproject.toml
```

---

## Further reading

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the system design and data-flow
  contract in depth.
- **[Architecture Decision Records](docs/decisions/)** — the *why* behind each
  major choice:
  - [0002 — Hybrid retrieval with RRF fusion](docs/decisions/0002-hybrid-retrieval-with-rrf-fusion.md)
  - [0003 — ChromaDB as vector store](docs/decisions/0003-chromadb-as-vector-store.md)
  - [0004 — Provider-agnostic LLM client](docs/decisions/0004-provider-agnostic-llm-client.md)
  - [0005 — API-free cross-encoder reranker](docs/decisions/0005-api-free-cross-encoder-reranker.md)
- **[docs/deploy-azure.md](docs/deploy-azure.md)** — cloud deployment walkthrough.

---

## Roadmap

Out of scope for v1, but natural next steps: streaming responses · conversation
memory · an evaluation harness (retrieval recall / answer faithfulness) ·
feedback collection · remote reranking (Cohere) · alternative vector stores
(Azure AI Search, pgvector) · multi-modal PDF parsing · a fully async pipeline.

---

## License

[MIT](LICENSE) © 2026 Joshua Quigley

## Disclaimer

This software is provided for educational and demonstration purposes only. It is
**not a medical device**, is **not intended for clinical decision-making**, and
must not be relied upon for patient care. It processes only public documents and
performs no PHI handling. Always consult primary sources and qualified
professionals.
