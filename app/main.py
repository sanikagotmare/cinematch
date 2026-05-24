from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.vector_store import get_vector_store
from app.models.embedder import get_embedder
from app.schemas.movie import HealthResponse
from app.services.recommendation import get_cache
from app.services.streaming import close_http_client

settings = get_settings()
setup_logging()

MOOD_GENRES = {
    "happy":       ["Comedy", "Animation", "Family", "Music"],
    "excited":     ["Action", "Adventure", "Thriller", "Science Fiction"],
    "sad":         ["Drama", "Romance"],
    "scared":      ["Horror", "Mystery", "Thriller"],
    "romantic":    ["Romance", "Drama"],
    "inspired":    ["Documentary", "History", "War"],
    "adventurous": ["Adventure", "Fantasy", "Science Fiction", "Western"],
    "relaxed":     ["Animation", "Family", "Comedy", "Documentary"],
    "thoughtful":  ["Drama", "Mystery", "Science Fiction", "History"],
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=" * 60)
    logger.info(f"  Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)
    logger.info("Loading embedding model (warm-up)...")
    get_embedder()
    logger.info("Connecting to ChromaDB vector store...")
    store = get_vector_store()
    logger.info(f"Vector store ready — {store.count()} movies indexed.")
    if store.count() == 0:
        logger.warning("Vector store is EMPTY. Run `python scripts/ingest.py` first.")
    logger.info("CineMatch API is ready to serve requests.")
    yield
    logger.info("Shutting down...")
    await close_http_client()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade Movie Recommendation API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ── API routes ─────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Mood endpoint ──────────────────────────────────────────────────────────
    @app.get("/api/v1/mood")
    async def mood_endpoint(
        mood: str = Query("happy"),
        genre: str = Query(None),
        top_k: int = Query(24),
    ):
        store = get_vector_store()
        genres = [genre] if genre else MOOD_GENRES.get(mood.lower(), ["Drama"])

        results = store._collection.get(
            include=["metadatas"],
            limit=5000,
        )
        metas = results["metadatas"] or []

        filtered = []
        for m in metas:
            mg = [g.strip().lower() for g in m.get("genres", "").split(",")]
            if any(g.lower() in mg for g in genres):
                filtered.append(m)

        filtered.sort(key=lambda x: float(x.get("vote_average", 0)), reverse=True)
        filtered = filtered[:top_k]

        output = []
        for m in filtered:
            # Use pre-built poster_url first, fallback to building from poster_path
            poster_url = m.get("poster_url", "")
            if not poster_url:
                poster_path = m.get("poster_path", "")
                if poster_path and not poster_path.startswith("http"):
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    poster_url = poster_path

            output.append({
                "movie_id":           int(m.get("movie_id", 0)),
                "title":              m.get("title", ""),
                "overview":           m.get("overview", ""),
                "genres":             [g.strip() for g in m.get("genres", "").split(",") if g.strip()],
                "vote_average":       float(m.get("vote_average", 0)),
                "vote_count":         int(m.get("vote_count", 0) or 0),
                "release_year":       str(m.get("release_year", "")),
                "poster_url":         poster_url,
                "director":           m.get("director", ""),
                "cast":               [c.strip() for c in m.get("cast", "").split(",") if c.strip()],
                "similarity_score":   float(m.get("vote_average", 0)) / 10.0,
                "streaming_providers": [],
            })
        return output

    # ── Watch Providers ────────────────────────────────────────────────────────
    @app.get("/api/v1/watch/{movie_id}")
    async def watch_providers(movie_id: int):
        import httpx
        api_key = settings.TMDB_API_KEY
        if not api_key or api_key == "YOUR_TMDB_API_KEY_HERE":
            return []
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
                    params={"api_key": api_key},
                )
                data = r.json()
                region = data.get("results", {}).get("IN", {})
                flatrate = region.get("flatrate", [])
                return [
                    {
                        "name":     p["provider_name"],
                        "logo":     f"https://image.tmdb.org/t/p/original{p.get('logo_path', '')}",
                        "priority": p.get("display_priority", 99),
                    }
                    for p in flatrate
                ]
        except Exception:
            return []

    # ── Health ─────────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Infra"])
    async def health() -> HealthResponse:
        store = get_vector_store()
        cache = get_cache()
        return HealthResponse(
            status="ok",
            version=settings.APP_VERSION,
            vector_store_docs=store.count(),
            cache_stats=cache.stats(),
        )

    # ── Serve Netflix UI ───────────────────────────────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()
