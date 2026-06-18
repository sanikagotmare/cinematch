"""
app/services/semantic_search.py
────────────────────────────────
Semantic search — "space horror" returns space horror movies even if
the words "space" or "horror" don't appear verbatim in the title.

How it works:
  The query string is embedded by the same transformer that was used
  during ingestion. Because all-MiniLM-L6-v2 was trained on sentence
  pairs, semantically related phrases land close together in the
  384-dimensional embedding space. "space horror" is geometrically
  near Alien, Event Horizon, Life, etc.

Interview talking point — why this beats TF-IDF:
  "TF-IDF is a bag-of-words model. 'Thrilling space film' would score
   zero similarity against a document that uses 'cosmic' and 'terrifying'
   because there are no shared tokens. A dense sentence embedding captures
   the latent meaning, not just the surface vocabulary."
"""
from __future__ import annotations

from loguru import logger

from app.db.vector_store import get_vector_store
from app.models.embedder import get_embedder
from app.schemas.movie import SearchResponse, SearchResult


async def semantic_search(
    query: str,
    top_k: int = 10,
) -> SearchResponse:
    """
    Embed the free-text query and return the top_k semantically
    closest movies from the vector store.
    """
    logger.info(f"Semantic search: '{query}' (top_k={top_k})")

    embedder = get_embedder()
    vector_store = get_vector_store()

    # Encode query — same model, same embedding space as the stored docs
    query_embedding = await embedder.async_encode(query)

    neighbours = vector_store.query_by_embedding(
        embedding=query_embedding[0].tolist(),
        top_k=top_k,
    )

    results: list[SearchResult] = []
    for n in neighbours:
        meta = n.metadata
        genres = [g.strip() for g in meta.get("genres", "").split(",") if g.strip()]
        poster_path = meta.get("poster_path", "")
        results.append(
            SearchResult(
                title=n.title,
                movie_id=n.movie_id,
                overview=meta.get("overview", ""),
                genres=genres,
                vote_average=float(meta.get("vote_average", 0.0)),
                release_year=str(meta.get("release_year", "")),
                relevance_score=n.similarity,
                poster_url = meta.get("poster_url", "") or (
            f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
        ),
            )
        )

    return SearchResponse(query=query, top_k=top_k, results=results)
