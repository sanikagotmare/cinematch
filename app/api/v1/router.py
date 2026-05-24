"""app/api/v1/router.py — Aggregates all v1 endpoints."""
from fastapi import APIRouter
from app.api.v1 import recommend, search

api_router = APIRouter()

api_router.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])
api_router.include_router(search.router,    prefix="/search",    tags=["Search"])
