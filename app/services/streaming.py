"""
app/services/streaming.py
──────────────────────────
Async TMDB Watch Providers integration via httpx.

This module demonstrates how a production system would fetch
real-time regional streaming availability (Netflix, Prime, etc.)
rather than hardcoding links.

Architecture notes for interviews:
  1. Connection pooling — a single shared AsyncClient is reused
     across all requests (keep-alive). Creating a new client per
     request would pay TCP + TLS handshake overhead every time.

  2. Graceful degradation — if TMDB is unavailable (timeout, 5xx),
     we return an empty list rather than propagating the error up.
     Streaming availability is enrichment, not core data — the
     recommendation response must always succeed.

  3. Structured error handling — we distinguish network errors from
     API errors from data-parsing errors and log each differently
     so on-call engineers can triage from dashboards.

  4. Region parameterisation — the TMDB_REGION env var lets you
     change the country without touching code. In a multi-tenant SaaS
     you would pass region per-request from the authenticated user's
     profile.
"""
from __future__ import annotations
import httpx
from loguru import logger

from app.core.config import get_settings
from app.schemas.movie import StreamingProvider

# ── Shared async HTTP client (connection pool) ────────────────────────────────
# Instantiated once at module import; reused for every request.
# Equivalent to a connection pool in a DB driver.
_settings = get_settings()

_http_client = httpx.AsyncClient(
    base_url=_settings.TMDB_BASE_URL,
    timeout=httpx.Timeout(
    timeout=10.0,
    connect=_settings.HTTP_CONNECT_TIMEOUT,
    read=_settings.HTTP_READ_TIMEOUT,
    write=5.0,
    pool=5.0,
),
    headers={"Accept": "application/json"},
    # In production, use TMDB v4 Bearer token instead of API key param
)


async def fetch_watch_providers(
    movie_id: int,
    region: str | None = None,
) -> list[StreamingProvider]:
    """
    Call TMDB /movie/{movie_id}/watch/providers and return a list of
    flatrate (subscription) streaming services for the given region.

    Returns an empty list on any error — callers must not depend on
    this data being present (graceful degradation pattern).

    TMDB endpoint docs:
      https://developer.themoviedb.org/reference/movie-watch-providers

    Example response structure:
      {
        "results": {
          "IN": {
            "flatrate": [
              {"provider_id": 8, "provider_name": "Netflix",
               "logo_path": "/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg",
               "display_priority": 1}
            ]
          }
        }
      }
    """
    region = region or _settings.TMDB_REGION

    # ── Guard: skip real call if no real API key is configured ────────────────
    if _settings.TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        logger.debug(
            f"TMDB_API_KEY not set — returning mock providers for movie_id={movie_id}"
        )
        return _mock_providers(movie_id)

    try:
        response = await _http_client.get(
            f"/movie/{movie_id}/watch/providers",
            params={"api_key": _settings.TMDB_API_KEY},
        )
        response.raise_for_status()
        data = response.json()

        region_data = data.get("results", {}).get(region, {})
        flatrate: list[dict] = region_data.get("flatrate", [])

        providers = [
            StreamingProvider(
                provider_name=p["provider_name"],
                logo_path=f"https://image.tmdb.org/t/p/original{p.get('logo_path', '')}",
                display_priority=p.get("display_priority", 99),
            )
            for p in flatrate
        ]
        logger.debug(
            f"Fetched {len(providers)} providers for movie_id={movie_id}, region={region}"
        )
        return providers

    except httpx.TimeoutException:
        logger.warning(f"TMDB timeout for movie_id={movie_id} — degrading gracefully")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"TMDB HTTP {exc.response.status_code} for movie_id={movie_id}"
        )
        return []
    except Exception as exc:
        logger.error(f"Unexpected error fetching providers for {movie_id}: {exc}")
        return []


async def close_http_client() -> None:
    """Called during app shutdown to drain the connection pool cleanly."""
    await _http_client.aclose()


# ── Mock fallback ─────────────────────────────────────────────────────────────

def _mock_providers(movie_id: int) -> list[StreamingProvider]:
    """
    Deterministic mock based on movie_id parity — purely for local
    development / demo purposes when no TMDB key is set.
    """
    if movie_id % 3 == 0:
        return [
            StreamingProvider(
                provider_name="Netflix",
                logo_path="https://image.tmdb.org/t/p/original/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg",
                display_priority=1,
            )
        ]
    if movie_id % 3 == 1:
        return [
            StreamingProvider(
                provider_name="Amazon Prime Video",
                logo_path="https://image.tmdb.org/t/p/original/emthp39XA2YScoYL1p0sdbAH2WA.jpg",
                display_priority=2,
            )
        ]
    return [
        StreamingProvider(
            provider_name="Apple TV+",
            logo_path="https://image.tmdb.org/t/p/original/6uhKBfmtzFqOcLousHwZuzcrScK.jpg",
            display_priority=3,
        )
    ]
