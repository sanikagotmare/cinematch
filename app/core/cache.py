"""
app/core/cache.py
─────────────────
Thread-safe, generic, in-memory TTL cache that mirrors the
semantics of a Redis GET / SETEX pair.

Design decisions worth discussing in an interview:
  1. Generic type parameter [T] — the cache is type-safe; callers
     know exactly what type they will receive back.
  2. Lazy eviction — expired entries are purged on every GET rather
     than a background thread, keeping the implementation dependency-free.
  3. Drop-in swap — replacing the dict with an actual Redis client
     (via `redis.asyncio`) requires changing only this file. Every
     service that depends on `CacheService` stays untouched.
  4. TTL reasoning — recommendation results are cached for 1 hour.
     Embedding queries are deterministic, so the result never changes
     unless the underlying vector store is rebuilt. This eliminates
     redundant ChromaDB round-trips on hot endpoints.

Cold-Start mitigation:
  On first request for a title the cache is cold → full vector
  query + TMDB HTTP call. Subsequent identical requests in the TTL
  window skip both and return in < 1 ms.
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float  # Unix timestamp


class TTLCache(Generic[T]):
    """
    A minimal, thread-safe in-memory cache with per-entry TTL.

    Usage:
        cache: TTLCache[list[MovieResponse]] = TTLCache(ttl=3600)
        cache.set("inception", results)
        hit = cache.get("inception")   # None if expired or missing
    """

    def __init__(self, ttl: int = 3600) -> None:
        self._ttl = ttl
        self._store: dict[str, _CacheEntry[T]] = {}
        self._lock = threading.Lock()

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[T]:
        """Return cached value or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]   # lazy eviction
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Store value with the configured TTL."""
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )

    def delete(self, key: str) -> None:
        """Manually invalidate a cache entry."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries (useful between test runs)."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, int]:
        """Return live count of total vs. expired entries."""
        now = time.monotonic()
        with self._lock:
            total = len(self._store)
            alive = sum(1 for e in self._store.values() if e.expires_at > now)
        return {"total": total, "alive": alive, "expired": total - alive}
