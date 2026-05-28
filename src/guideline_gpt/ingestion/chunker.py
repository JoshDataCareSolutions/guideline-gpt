"""Split page text into token-bounded chunks with stable identity.

Chunking uses LlamaIndex's :class:`SentenceSplitter`, configured with the
token-based ``chunk_size``/``chunk_overlap`` from :class:`Settings`. Token counts
use the ``cl100k_base`` encoding (tiktoken), matching the OpenAI embedding model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from functools import lru_cache

import tiktoken
from llama_index.core.node_parser import SentenceSplitter

from guideline_gpt.config import Settings
from guideline_gpt.logging_setup import get_logger
from guideline_gpt.types import Chunk, DocumentMetadata

log = get_logger(__name__)

_CHUNK_ID_LENGTH = 16


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Return the cached cl100k_base tiktoken encoding."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` using the cl100k_base encoding."""
    return len(_encoding().encode(text))


def make_chunk_id(source_path: str, page_number: int, text: str) -> str:
    """Compute a stable chunk id: first 16 hex chars of SHA-256(source|page|text).

    The id is deterministic, so re-ingesting an unchanged corpus produces the
    same ids (enabling idempotent upserts).
    """
    digest = hashlib.sha256(f"{source_path}|{page_number}|{text}".encode())
    return digest.hexdigest()[:_CHUNK_ID_LENGTH]


def _make_splitter(settings: Settings) -> SentenceSplitter:
    """Build a SentenceSplitter that measures size in cl100k_base tokens."""
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        tokenizer=_encoding().encode,
    )


def chunk_pages(
    pages: Iterable[tuple[str, DocumentMetadata]],
    settings: Settings,
) -> list[Chunk]:
    """Chunk an iterable of (page_text, metadata) pairs into :class:`Chunk` objects.

    Page-number metadata is preserved through chunking; every chunk derived from
    a page carries that page's :class:`DocumentMetadata`.

    Args:
        pages: Iterable of (page_text, metadata) pairs (e.g. from ``load_pdfs``).
        settings: Configuration providing chunk size and overlap.

    Returns:
        A list of chunks in document order.
    """
    splitter = _make_splitter(settings)
    chunks: list[Chunk] = []

    for text, metadata in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(metadata.source_path, metadata.page_number, piece),
                    text=piece,
                    metadata=metadata,
                    token_count=count_tokens(piece),
                )
            )

    log.info("chunked", chunks=len(chunks))
    return chunks
