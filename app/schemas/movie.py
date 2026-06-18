"""
app/schemas/movie.py
─────────────────────
Pydantic v2 schemas for every API boundary.

Why this matters in an interview:
  "Pydantic schemas act as a compile-time contract between the client
   and the service layer. Any field mismatch raises a 422 Unprocessable
   Entity with a human-readable explanation before business logic runs."
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ── Streaming provider (inner object) ────────────────────────────────────────

class StreamingProvider(BaseModel):
    provider_name: str
    logo_path: Optional[str] = None
    display_priority: int = 99

    model_config = ConfigDict(frozen=True)


# ── Individual recommendation item ───────────────────────────────────────────

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    overview: str
    genres: list[str]
    vote_average: float = Field(..., ge=0.0, le=10.0)
    release_year: str
    similarity_score: float = Field(..., ge=0.0, le=1.0,
        description="Cosine similarity in [0, 1]; higher is more similar")
    poster_url: Optional[str] = None
    streaming_providers: list[StreamingProvider] = Field(
        default_factory=list,
        description="Real-time regional watch providers from TMDB"
    )

    model_config = ConfigDict(frozen=True)


# ── /recommend/{movie_title} response ────────────────────────────────────────

class RecommendationResponse(BaseModel):
    query_title: str
    top_k: int
    cached: bool = Field(False,
        description="True when the result was served from in-memory cache")
    recommendations: list[MovieRecommendation]

    model_config = ConfigDict(frozen=True)


# ── /search?q= response ──────────────────────────────────────────────────────

class SearchResult(BaseModel):
    title: str
    movie_id: int
    overview: str
    genres: list[str]
    vote_average: float = Field(..., ge=0.0, le=10.0)
    release_year: str
    relevance_score: float = Field(..., ge=0.0, le=1.0,
        description="Semantic similarity to the query string")
    poster_url: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]

    model_config = ConfigDict(frozen=True)


# ── Health-check response ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_docs: int
    cache_stats: dict[str, int]

    model_config = ConfigDict(frozen=True)
