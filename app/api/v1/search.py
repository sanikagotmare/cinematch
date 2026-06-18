"""
app/api/v1/search.py
─────────────────────
GET /search?q=space+horror
"""
from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.core.config import get_settings
from app.schemas.movie import SearchResponse
from app.services.semantic_search import semantic_search

router = APIRouter()
settings = get_settings()


@router.get(
    "",
    response_model=SearchResponse,
    summary="Semantic movie search",
    description=(
        "Embed the free-text query with the same transformer used during "
        "ingestion and return the semantically closest movies. "
        "Searching 'lonely astronaut survival' will surface films like "
        "Gravity and The Martian even without matching title keywords."
    ),
)
async def search(
    q: Annotated[
        str,
        Query(
            title="Search query",
            description="Natural-language description of the kind of movie you want",
            min_length=2,
            max_length=300,
            example="space horror claustrophobic",
        ),
    ],
    top_k: Annotated[
        int,
        Query(ge=1, le=settings.MAX_TOP_K),
    ] = 10,
) -> SearchResponse:
    try:
        return await semantic_search(query=q, top_k=top_k)
    except Exception as exc:
        logger.exception(f"Search failed for query '{q}': {exc}")
        raise HTTPException(
            status_code=500,
            detail="Search service encountered an error. Please try again.",
        ) from exc
