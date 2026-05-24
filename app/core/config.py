"""
app/core/config.py
──────────────────
Single source of truth for every environment variable.
Pydantic-Settings reads from the .env file automatically;
all fields are type-validated at startup so misconfiguration
fails fast — before a single request is served.

Interview talking point:
  "Centralising config in a validated Settings object means you
   can never silently swallow a missing API key at runtime — the
   process dies at import time with a clear validation error."
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # ── App meta ─────────────────────────────────────────────────────────────
    APP_NAME: str = "CineMatch API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    MOVIES_CSV: str = "data/tmdb_5000_movies.csv"
    CREDITS_CSV: str = "data/tmdb_5000_credits.csv"
    CHROMA_PERSIST_DIR: str = ".chromadb"
    CHROMA_COLLECTION: str = "movies"

    # ── Embedding model ───────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 → 384-dim embeddings, ~80 MB, runs well on CPU
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Recommendation defaults ───────────────────────────────────────────────
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    # ── Cache (in-memory Redis mock) ──────────────────────────────────────────
    CACHE_TTL_SECONDS: int = 3600   # 1 hour; swap for redis:// URI in prod

    # ── TMDB API ──────────────────────────────────────────────────────────────
    TMDB_API_KEY: str = "YOUR_TMDB_API_KEY_HERE"
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_REGION: str = "IN"         # ISO 3166-1 country code for watch providers

    # ── HTTP client timeouts (seconds) ────────────────────────────────────────
    HTTP_CONNECT_TIMEOUT: float = 5.0
    HTTP_READ_TIMEOUT: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton — same object is returned on every call.
    Use as a FastAPI dependency: Depends(get_settings)
    """
    return Settings()
