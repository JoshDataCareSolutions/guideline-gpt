"""Ingestion orchestrator: PDFs -> chunks -> ChromaDB + BM25 index.

Re-running is idempotent: chunk ids are deterministic, so the Chroma collection
is upserted (not duplicated). Chunks whose source document was removed or edited
are pruned from the collection before upsert, so the vector store stays in sync
with the freshly rebuilt BM25 index rather than drifting.
"""

from __future__ import annotations

import time
from pathlib import Path

from guideline_gpt.config import Settings
from guideline_gpt.ingestion.chunker import chunk_pages
from guideline_gpt.ingestion.loader import load_pdfs
from guideline_gpt.logging_setup import get_logger
from guideline_gpt.retrieval.bm25 import build_bm25, save_bm25
from guideline_gpt.retrieval.vector import chunk_to_record, get_collection
from guideline_gpt.types import Chunk, IngestionReport

log = get_logger(__name__)

# Upsert in batches so the embedding API isn't sent the whole corpus at once.
_UPSERT_BATCH = 100


class IngestionError(RuntimeError):
    """Raised when ingestion produces no usable chunks."""


def _sync_chunks(settings: Settings, chunks: list[Chunk]) -> None:
    """Reconcile the Chroma collection with the current corpus, then upsert.

    Chroma ``upsert`` adds and updates but never removes, so a chunk whose source
    PDF was deleted (or edited) would linger in the vector store and drift from
    the freshly rebuilt BM25 index. We first delete any stored id that is no
    longer in the current chunk set, keeping the two stores consistent and
    re-ingestion truly idempotent.
    """
    collection = get_collection(settings)

    current_ids = {chunk.chunk_id for chunk in chunks}
    existing_ids = set(collection.get(include=[])["ids"])
    orphaned = list(existing_ids - current_ids)
    if orphaned:
        collection.delete(ids=orphaned)
        log.info("pruned_orphans", count=len(orphaned))

    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        ids: list[str] = []
        documents: list[str] = []
        metadatas = []
        for chunk in batch:
            cid, doc, meta = chunk_to_record(chunk)
            ids.append(cid)
            documents.append(doc)
            metadatas.append(meta)
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        log.info("upserted_batch", count=len(batch), done=start + len(batch), total=len(chunks))


def ingest(documents_dir: Path, settings: Settings) -> IngestionReport:
    """Ingest every PDF in ``documents_dir`` into the vector and BM25 indexes.

    Args:
        documents_dir: Folder of PDFs to ingest.
        settings: Configuration (storage paths, chunk sizes, embedding model).

    Returns:
        An :class:`IngestionReport` summarizing the run.

    Raises:
        IngestionError: If no chunks were produced from the corpus.
    """
    started = time.perf_counter()
    log.info("ingest_start", documents_dir=str(documents_dir))

    pages = list(
        load_pdfs(documents_dir, boilerplate_page_fraction=settings.boilerplate_page_fraction)
    )
    chunks = chunk_pages(pages, settings)
    if not chunks:
        raise IngestionError(
            f"No chunks produced from {documents_dir}. Are there readable PDFs there?"
        )

    _sync_chunks(settings, chunks)

    bm25, chunk_ids = build_bm25(chunks)
    save_bm25(settings.bm25_path, bm25, chunk_ids)

    # Distinct source files among the kept pages.
    file_count = len({meta.source_path for _, meta in pages})
    total_tokens = sum(chunk.token_count for chunk in chunks)
    elapsed = time.perf_counter() - started

    report = IngestionReport(
        files=file_count,
        pages=len(pages),
        chunks=len(chunks),
        total_tokens=total_tokens,
        elapsed_seconds=round(elapsed, 2),
    )
    log.info(
        "ingest_done",
        files=report.files,
        pages=report.pages,
        chunks=report.chunks,
        total_tokens=report.total_tokens,
        elapsed_seconds=report.elapsed_seconds,
    )
    return report
