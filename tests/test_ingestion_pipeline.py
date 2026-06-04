"""Tests for ingestion's Chroma membership reconciliation (mocked collection).

The contract under test: ``_sync_chunks`` must delete any stored chunk id that
is no longer in the current corpus before upserting, so the vector store does
not drift from the freshly rebuilt BM25 index (see ADR-003).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from guideline_gpt.config import Settings
from guideline_gpt.ingestion import pipeline as pipeline_module
from guideline_gpt.ingestion.pipeline import _sync_chunks
from guideline_gpt.types import Chunk, DocumentMetadata


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        metadata=DocumentMetadata(
            source_path="/corpus/g.pdf",
            source_name="g",
            page_number=1,
        ),
        token_count=3,
    )


def _fake_collection(existing_ids: list[str]) -> MagicMock:
    collection = MagicMock()
    collection.get.return_value = {"ids": existing_ids}
    return collection


def test_sync_prunes_orphaned_chunks(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Store had a, b, c; the current corpus only has a, b -> c is an orphan.
    collection = _fake_collection(["a", "b", "c"])
    monkeypatch.setattr(pipeline_module, "get_collection", lambda _s: collection)

    _sync_chunks(settings, [_chunk("a"), _chunk("b")])

    collection.delete.assert_called_once()
    deleted_ids = collection.delete.call_args.kwargs["ids"]
    assert set(deleted_ids) == {"c"}

    # Current chunks are still upserted after pruning.
    collection.upsert.assert_called_once()
    assert set(collection.upsert.call_args.kwargs["ids"]) == {"a", "b"}


def test_sync_without_orphans_does_not_delete(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Store and corpus agree exactly -> nothing to prune.
    collection = _fake_collection(["a", "b"])
    monkeypatch.setattr(pipeline_module, "get_collection", lambda _s: collection)

    _sync_chunks(settings, [_chunk("a"), _chunk("b")])

    collection.delete.assert_not_called()
    collection.upsert.assert_called_once()


def test_sync_into_empty_collection_does_not_delete(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First-ever ingest: empty store, so no delete call (delete([]) is wasteful).
    collection = _fake_collection([])
    monkeypatch.setattr(pipeline_module, "get_collection", lambda _s: collection)

    _sync_chunks(settings, [_chunk("a")])

    collection.delete.assert_not_called()
    collection.upsert.assert_called_once()
