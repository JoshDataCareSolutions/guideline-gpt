"""ChromaDB vector store: collection factory, record (de)serialization, search.

The collection is created with an OpenAI embedding function so that both
ingestion (adding documents) and querying (embedding the query) use the same
configured embedding model. Search is added in :class:`VectorRetriever`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions

from guideline_gpt.config import Settings
from guideline_gpt.types import Chunk, DocumentMetadata, RetrievalHit

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


def _embedding_function(settings: Settings) -> embedding_functions.EmbeddingFunction:
    """Build the OpenAI embedding function from settings.

    Raises:
        ValueError: If no OpenAI API key is configured (always required).
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings (regardless of LLM_PROVIDER).")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


def get_collection(settings: Settings) -> Collection:
    """Open (or create) the persistent Chroma collection for the corpus.

    Cosine distance is used so that scores can be normalized to ``[0, 1]``.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=_embedding_function(settings),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_to_record(chunk: Chunk) -> tuple[str, str, Metadata]:
    """Serialize a chunk into (id, document, metadata) for Chroma storage."""
    meta: Metadata = {
        "source_path": chunk.metadata.source_path,
        "source_name": chunk.metadata.source_name,
        "page_number": chunk.metadata.page_number,
        "section_heading": chunk.metadata.section_heading or "",
        "token_count": chunk.token_count,
    }
    return chunk.chunk_id, chunk.text, meta


def record_to_chunk(chunk_id: str, document: str, metadata: Metadata) -> Chunk:
    """Reconstruct a :class:`Chunk` from a stored Chroma record."""
    heading = str(metadata.get("section_heading") or "") or None
    return Chunk(
        chunk_id=chunk_id,
        text=document,
        metadata=DocumentMetadata(
            source_path=str(metadata["source_path"]),
            source_name=str(metadata["source_name"]),
            page_number=int(str(metadata["page_number"])),
            section_heading=heading,
        ),
        token_count=int(str(metadata["token_count"])),
    )


class VectorRetriever:
    """Dense retrieval over the Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        self._collection = get_collection(settings)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return the top-k nearest chunks to ``query`` by cosine similarity.

        Cosine distance in ``[0, 2]`` is converted to a similarity score in
        ``[0, 1]`` via ``1 - distance / 2``.

        Args:
            query: The natural-language query.
            top_k: Number of hits to return.

        Returns:
            Ranked retrieval hits (rank is 1-indexed).
        """
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        hits: list[RetrievalHit] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            hits.append(
                RetrievalHit(
                    chunk=record_to_chunk(cid, doc, meta),
                    score=max(0.0, min(1.0, 1.0 - float(dist) / 2.0)),
                    source="vector",
                    rank=rank,
                )
            )
        return hits
