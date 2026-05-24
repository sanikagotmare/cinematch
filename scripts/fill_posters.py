"""
scripts/fill_posters.py
────────────────────────
Run this to fill in missing poster URLs without re-ingesting everything.
Only updates movies that currently have empty poster_url in ChromaDB.

Usage:
    set TMDB_API_KEY=your_key_here
    python scripts/fill_posters.py
"""
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.db.vector_store import get_vector_store

settings = get_settings()


def fetch_poster(movie_id: int, api_key: str) -> str:
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key},
            timeout=5,
        )
        if r.status_code == 200:
            path = r.json().get("poster_path", "")
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return ""


def main():
    api_key = settings.TMDB_API_KEY
    if not api_key or api_key == "YOUR_TMDB_API_KEY_HERE":
        print("ERROR: Set TMDB_API_KEY in .env or via set command first!")
        return

    print("Connecting to ChromaDB...")
    vs = get_vector_store()

    print("Fetching all records...")
    results = vs._collection.get(include=["metadatas"])
    ids     = results["ids"]
    metas   = results["metadatas"]

    # Find movies missing poster_url
    missing = [
        (chroma_id, meta)
        for chroma_id, meta in zip(ids, metas)
        if not meta.get("poster_url", "").strip()
    ]
    print(f"Found {len(missing):,} movies missing posters out of {len(ids):,} total.")

    if not missing:
        print("All movies already have posters!")
        return

    filled  = 0
    failed  = 0
    t0      = time.perf_counter()

    for i, (chroma_id, meta) in enumerate(missing):
        movie_id = int(meta.get("movie_id", 0))
        if not movie_id:
            continue

        poster_url = fetch_poster(movie_id, api_key)
        if poster_url:
            # Update just this record's metadata
            updated_meta = {**meta, "poster_url": poster_url}
            vs._collection.update(
                ids=[chroma_id],
                metadatas=[updated_meta],
            )
            filled += 1
        else:
            failed += 1

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i+1}/{len(missing)} — "
                f"filled: {filled} — "
                f"ETA: {remaining/60:.1f} min",
                end="\r"
            )

        # Small delay to respect TMDB rate limit (40 req/10s)
        time.sleep(0.26)

    elapsed = time.perf_counter() - t0
    print(f"\n\nDone in {elapsed/60:.1f} min")
    print(f"  Filled: {filled:,} posters")
    print(f"  Failed: {failed:,} (movie not found on TMDB)")
    print(f"\nRestart uvicorn to see all posters: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
