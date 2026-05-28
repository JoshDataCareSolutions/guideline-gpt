# guideline-gpt

[![CI](https://github.com/joshquigley/guideline-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/joshquigley/guideline-gpt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

A production-pattern **Retrieval-Augmented Generation (RAG)** system for
querying domain-specific guideline corpora — with a **pipeline inspector UI**
that shows you *how the system works* at every stage, not just a chat box.

The default showcase corpus is public clinical guidelines (respiratory and
critical care), but the system is **corpus-agnostic**: drop your own PDFs into
`documents/`, rebuild the index, and query your domain.

> ⚠️ **Not a medical device.** This project is for education and engineering
> demonstration only — not for clinical decision-making. Public documents only;
> no PHI handling. See the disclaimer below.

---

## Why this exists

Most RAG demos are a single chat box that hides everything interesting. This one
exposes the full pipeline so you can *see* retrieval, fusion, reranking, prompt
assembly, and generation — including token counts and per-query cost.

```
Ingest (offline):  PDFs → parse → chunk → embed → ChromaDB (+ BM25 index)
Query (online):    query → vector + BM25 search → RRF fusion → cross-encoder
                   rerank → prompt assembly → LLM → cited answer + full trace
```

The **`QueryTrace`** object captures every stage; the inspector UI is just a
rendering of it.

---

## Quickstart (local, ~5 minutes)

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if you don't have it, then:
uv pip install -e ".[dev]"

# 2. Configure
cp .env.example .env        # add your OPENAI_API_KEY (+ ANTHROPIC_API_KEY)

# 3. Add documents and build the index
python scripts/download_corpus.py   # or drop your own PDFs in documents/
guideline-gpt ingest

# 4. Launch the inspector UI
guideline-gpt serve
```

### Docker (1 command)

```bash
docker compose up
```

---

## Configuration

All configuration is via environment variables, validated by Pydantic. See
[`.env.example`](.env.example) for the full list. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Default Anthropic model |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RETRIEVAL_TOP_K` | `20` | Candidates before rerank |
| `RERANK_TOP_K` | `5` | Chunks sent to the LLM |

`OPENAI_API_KEY` is always required (it powers embeddings) regardless of the
chosen LLM provider.

---

## Development

```bash
uv pip install -e ".[dev]"
ruff check src tests          # lint
ruff format src tests         # format
mypy src                      # strict type check
pytest --cov=guideline_gpt    # tests + coverage
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [docs/deploy-azure.md](docs/deploy-azure.md) — Azure Container Apps deploy
- [docs/decisions/](docs/decisions/) — Architecture Decision Records

---

## Roadmap (out of scope for v1)

Streaming responses · multi-modal PDFs · conversation memory · feedback
collection · eval harness (companion project) · remote rerank (Cohere) ·
alternative vector stores (Azure AI Search, pgvector) · async pipeline.

---

## License

[MIT](LICENSE) © 2026 Joshua Quigley

## ⚠️ Disclaimer

This software is provided for educational and demonstration purposes only. It is
**not a medical device**, is **not intended for clinical decision-making**, and
must not be relied upon for patient care. It processes only public documents and
performs no PHI handling. Always consult primary sources and qualified
professionals.
