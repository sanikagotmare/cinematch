"""
app/db/vector_store.py
───────────────────────
ChromaDB client wrapper.

ChromaDB stores:
  • Embeddings (384-dim float32 vectors from all-MiniLM-L6-v2)
  • Documents  (the combined text string used to generate the embedding)
  • Metadata   (title, genres, vote_average, release_year, movie_id,
                poster_path — everything the API response needs)

Query pattern:
  cosine similarity search returns the N nearest neighbours along
  with their distance scores (0 = identical, 2 = orthogonal for
  normalised vectors). We convert to similarity = 1 - distance/2
  so the value lands in [0, 1].

Interview talking points:
  1. "ChromaDB persists its index to disk, so the embedding matrix
     survives a process restart without re-running the O(n) ingestion
     pipeline — addressing the cold start problem at the infrastructure
     level, not just in application code."

  2. "The collection is created once and reused. The `get_or_create`
     pattern means the ingestion script and the API server can both
     call this function safely without race conditions."

  3. "Switching to a hosted vector DB (Pinecone, Weaviate, Qdrant) in
     production requires changing only this file — the service layer
     is fully decoupled from the storage backend."
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.core.config import get_settings


@dataclass(frozen=True)
class VectorQueryResult:
    """Typed result from a vector similarity query."""
    movie_id: int
    title: str
    document: str          # original combined-text used for embedding
    metadata: dict[str, Any]
    similarity: float      # cosine similarity in [0, 1]


class VectorStore:
    """Async-safe ChromaDB collection wrapper."""

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_PERSIST_DIR}")

        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # cosine distance metric
        )
        logger.info(
            f"ChromaDB collection '{settings.CHROMA_COLLECTION}' ready — "
            f"{self._collection.count()} documents indexed."
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Upsert a batch of movie embeddings.
        'upsert' is idempotent — safe to call multiple times on the
        same IDs (e.g., when re-running the ingestion script).
        """
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        exclude_id: str | None = None,
    ) -> list[VectorQueryResult]:
        """
        Find the top-k nearest neighbours to the given embedding vector.

        exclude_id: when recommending similar movies, exclude the query
                    movie itself from results.
        """
        where_filter = {"movie_id": {"$ne": int(exclude_id)}} if exclude_id else None

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k + (1 if exclude_id else 0),  # fetch +1 to absorb self-match
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        output: list[VectorQueryResult] = []
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for _id, doc, meta, dist in zip(ids, docs, metas, distances):
            if exclude_id and _id == exclude_id:
                continue
            # ChromaDB cosine space → distance in [0, 2]; map to similarity [0, 1]
            similarity = round(max(0.0, 1.0 - dist / 2.0), 4)
            output.append(
                VectorQueryResult(
                    movie_id=int(meta.get("movie_id", 0)),
                    title=meta.get("title", ""),
                    document=doc,
                    metadata=meta,
                    similarity=similarity,
                )
            )
        return output[:top_k]

    def get_by_title(self, title: str) -> VectorQueryResult | None:
        """
        Exact-match title lookup — used to retrieve a movie's own
        pre-computed embedding so we avoid re-encoding known titles.

        This is the key latency win: for known movies we never call
        the transformer at query time.
        """
        results = self._collection.get(
            where={"title": title},
            include=["embeddings", "documents", "metadatas"],
            limit=1,
        )
        if not results["ids"]:
            return None

        meta = results["metadatas"][0]
        return VectorQueryResult(
            movie_id=int(meta.get("movie_id", 0)),
            title=meta.get("title", ""),
            document=results["documents"][0],
            metadata=meta,
            similarity=1.0,   # perfect match with itself
        )

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Singleton — same ChromaDB client for the process lifetime."""
    return VectorStore()
