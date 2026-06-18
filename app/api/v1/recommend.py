"""
app/api/v1/recommend.py
────────────────────────
GET /recommend/{movie_title}
"""
from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Path
from loguru import logger

from app.core.config import get_settings
from app.schemas.movie import RecommendationResponse
from app.services.recommendation import get_recommendations

router = APIRouter()
settings = get_settings()


@router.get(
    "/{movie_title}",
    response_model=RecommendationResponse,
    summary="Get movie recommendations",
    description=(
        "Returns the top-k movies most similar to the given title "
        "using dense vector embeddings and cosine similarity search. "
        "Results include real-time streaming provider availability. "
        "Responses are cached in-memory for 1 hour."
    ),
)
async def recommend(
    movie_title: Annotated[
        str,
        Path(
            title="Movie title",
            description="Exact or approximate title of the seed movie",
            min_length=1,
            max_length=200,
            example="Inception",
        ),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=settings.MAX_TOP_K,
            description=f"Number of recommendations to return (max {settings.MAX_TOP_K})",
        ),
    ] = settings.DEFAULT_TOP_K,
) -> RecommendationResponse:
    try:
        return await get_recommendations(movie_title=movie_title, top_k=top_k)
    except Exception as exc:
        logger.exception(f"Recommendation failed for '{movie_title}': {exc}")
        raise HTTPException(
            status_code=500,
            detail="Recommendation service encountered an error. Please try again.",
        ) from exc
