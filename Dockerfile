# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   - builder: uv-managed virtualenv with all runtime deps installed
#   - runtime: slim Python image with the venv + source; runs as non-root.
#
# Image size note: sentence-transformers pulls torch (~700MB on CPU), so the
# final image is large. That's an explicit tradeoff for having an offline,
# API-free reranker — see ADR-005.

# ---------- builder ----------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Copy only what's needed to resolve the dependency set first, then the source,
# so dep changes don't bust the source cache and vice versa.
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /app/.venv \
 && uv pip install --python /app/.venv/bin/python -e .

# ---------- runtime ----------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp1: required by torch / sentence-transformers at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ARG UID=10001
RUN useradd --uid ${UID} --create-home --shell /bin/bash app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src

# Writable mount points for the corpus and vector store.
RUN mkdir -p /app/documents /app/chroma_db \
 && chown -R app:app /app/documents /app/chroma_db

USER app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    DOCUMENTS_DIR=/app/documents \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "src/guideline_gpt/ui/streamlit_app.py"]
