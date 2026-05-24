from __future__ import annotations
import ast
import sys
import time
import requests
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.db.vector_store import get_vector_store
from app.models.embedder import get_embedder

settings = get_settings()
BATCH_SIZE = 64

# ── TMDB poster fetcher ────────────────────────────────────────────────────────
# We use the TMDB API to get poster paths in bulk.
# If no API key, we use a deterministic placeholder color per movie.
TMDB_API_KEY = settings.TMDB_API_KEY if settings.TMDB_API_KEY != "YOUR_TMDB_API_KEY_HERE" else None

def fetch_poster_url(movie_id: int, title: str = "") -> str:
    """Fetch poster URL from TMDB API using movie ID."""
    if TMDB_API_KEY:
        try:
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                poster = data.get("poster_path", "")
                if poster:
                    return f"https://image.tmdb.org/t/p/w500{poster}"
        except Exception:
            pass
    return ""


def fetch_posters_bulk(movie_ids: list, titles: list) -> dict:
    """
    Fetch poster URLs for all movies.
    With API key: real TMDB posters.
    Without API key: use TMDB image CDN with known poster paths
    from a pre-built mapping for top movies, empty string otherwise.
    """
    # Known poster paths for top TMDB movies (works without API key)
    KNOWN_POSTERS = {
        19995: "/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",   # Avatar
        285:   "/9O7gLzmreU0nGkIB6K3BsJbzvNv.jpg",   # Pirates
        206647:"/cezWGskPY5x7GaglTTRN4Fugfb8.jpg",  # Spectre
        49026: "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",   # Dark Knight Rises
        49529: "/2oMtzHhF8ubS5LGllNnJC0HLFXM.jpg",  # Spider-Man 3
        559:   "/bFldqQ3eDLGStoHNS4UNgF6lOhP.jpg",   # Spider-Man
        38757: "/4m1Au3YkjqsxF8iwQy0fPYSxE0h.jpg",  # Tangled
        807:   "/kvXLZqY0Ngl1XSaJWm3NONtLnJg.jpg",   # Se7en
        703:   "/ugE2BKIGG6Oh9AUSpHhiCWqOvCI.jpg",   # Four Brothers
        857:   "/d50mP7HqBSGVOsUDKsOhFBnFpCy.jpg",   # Saving Private Ryan
        27205: "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",   # Inception
        157336:"/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",  # Interstellar
        155:   "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",   # Dark Knight
        122:   "/5VTN0pR8gcqV3EPUHHfMGnJYspq.jpg",   # LOTR Return
        680:   "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",   # Pulp Fiction
        13:    "/clolk7rB11OjR0p4WEAnXCEQYHT.jpg",   # Forrest Gump
        550:   "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",   # Fight Club
        238:   "/3bhkrj58Vtu7enYsLagi1rp6k6R.jpg",   # Godfather
        424:   "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",   # Schindler's List
        129:   "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",   # Spirited Away
        128:   "/oFaHRdlTTqDdSPeaBwMGJaJGPOM.jpg",   # Princess Mononoke
        4935:  "/6GMHlPuWKHWr5HWpTQKTPi7pQGg.jpg",   # Howl's Moving Castle
        49521: "/3iYQTLGoy7QnjcNYam5Au9SXKFT.jpg",   # Kung Fu Panda 2
        10193: "/WLQN5aiQG8wc9SeKwixW7pAR8K.jpg",    # Toy Story 3
        862:   "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",   # Toy Story
        863:   "/8oBHDMOgNinjCIQL8LyvN0WxOBG.jpg",   # Toy Story 2
        597:   "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",   # Titanic
        120:   "/56zTpe2xvadfxfj5mHp2PADMbK5.jpg",   # LOTR Fellowship
        121:   "/5c0ovjT41bJgQTR6LEJQE2AKGFQ.jpg",   # LOTR Two Towers
        11:    "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",   # Star Wars
        1891:  "/lWjHjNKRJjhRNqnJQCRbIkNnrTH.jpg",   # Empire Strikes Back
        1892:  "/mDCBQNhR6R0PVFucJp8LMZMvCBh.jpg",   # Return of the Jedi
        76341: "/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",   # Mad Max Fury Road
        72190: "/6Gwmh9sHQPBVT5HjhzHCfRJmGlH.jpg",   # Thor Dark World
        76338: "/5WJNpyFNHXanVSjAMbgxOlI5Ckr.jpg",   # Thor Ragnarok
        101:   "/yI6X2cCM5YPJtxMhUd3dPGqDAhw.jpg",   # Leon Professional
        280:   "/5Et7PGHGkEE0xHJaH5frO0PGDDE.jpg",   # Terminator 2
        218:   "/7XzowxZOv4WHHD1RtReQ8yjIGhd.jpg",   # Shining
        11324: "/aPG0906HRzEzJlL6lMkfQOaOLMb.jpg",   # Shutter Island
    }

    result = {}
    if TMDB_API_KEY:
        print(f"\n    Fetching posters from TMDB API (this may take a few minutes)...")
        for i, (mid, title) in enumerate(zip(movie_ids, titles)):
            result[mid] = fetch_poster_url(mid, title)
            if i % 100 == 0:
                print(f"    Fetched {i}/{len(movie_ids)} posters...", end="\r")
        print(f"\n    Fetched {len(result)} posters.")
    else:
        print("\n    No TMDB API key — using known poster map for top movies.")
        print("    Tip: Add TMDB_API_KEY to .env for full poster coverage!")
        for mid in movie_ids:
            path = KNOWN_POSTERS.get(mid, "")
            result[mid] = f"https://image.tmdb.org/t/p/w500{path}" if path else ""

    return result


def _parse_names(value, key="name", limit=None):
    try:
        items = ast.literal_eval(value)
        names = [i[key] for i in items if isinstance(i, dict) and key in i]
        return names[:limit] if limit else names
    except Exception:
        return []


def _parse_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        return next(
            (c["name"] for c in crew if isinstance(c, dict) and c.get("job") == "Director"),
            "",
        )
    except Exception:
        return ""


def _build_document(row):
    genres   = " ".join(row["genre_list"])
    keywords = " ".join(row["keyword_list"][:15])
    cast     = " ".join(row["cast_list"][:5])
    director = row["director"]
    overview = row["overview"] or ""
    return f"{overview} {genres} {genres} {genres} {keywords} {keywords} {cast} {director}".strip()


def main():
    t0 = time.perf_counter()
    print("-" * 60)
    print("  CineMatch Ingestion Pipeline")
    print("-" * 60)

    print(f"\n[1/5] Loading CSVs...")
    movies_path  = Path(settings.MOVIES_CSV)
    credits_path = Path(settings.CREDITS_CSV)

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing: {movies_path}")
    if not credits_path.exists():
        raise FileNotFoundError(f"Missing: {credits_path}")

    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    print(f"    Loaded {len(movies):,} movies and {len(credits):,} credit records.")

    print("\n[2/5] Merging and parsing columns...")
    df = movies.merge(credits[["title", "cast", "crew"]], on="title", how="left")

    df["genre_list"]   = df["genres"].fillna("[]").apply(_parse_names)
    df["keyword_list"] = df["keywords"].fillna("[]").apply(_parse_names)
    df["cast_list"]    = df["cast"].fillna("[]").apply(lambda x: _parse_names(x, limit=5))
    df["director"]     = df["crew"].fillna("[]").apply(_parse_director)
    df["overview"]     = df["overview"].fillna("")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["release_year"] = df["release_date"].astype(str).str[:4]
    df["tagline"]      = df.get("tagline", pd.Series([""] * len(df))).fillna("")

    df = df[
        (df["title"].notna()) &
        (df["vote_count"] >= 20) &
        (df["overview"].str.len() > 20)
    ].reset_index(drop=True)
    print(f"    {len(df):,} movies after quality filter.")

    print("\n[3/5] Building documents...")
    df["document"] = df.apply(_build_document, axis=1)

    print("\n[4/5] Fetching poster URLs...")
    movie_ids  = df["id"].astype(int).tolist()
    titles     = df["title"].tolist()
    poster_map = fetch_posters_bulk(movie_ids, titles)
    df["poster_url"] = df["id"].apply(lambda x: poster_map.get(int(x), ""))
    filled = df["poster_url"].astype(bool).sum()
    print(f"    {filled:,}/{len(df):,} movies have poster URLs.")

    print(f"\n[5/5] Generating embeddings (batch_size={BATCH_SIZE})...")
    embedder  = get_embedder()
    documents = df["document"].tolist()
    all_embeddings = []
    total_batches  = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i: i + BATCH_SIZE]
        embs  = embedder.encode(batch, normalize=True)
        all_embeddings.append(embs)
        print(f"    Batch {i // BATCH_SIZE + 1}/{total_batches}", end="\r")

    matrix = np.vstack(all_embeddings)
    print(f"\n    Done. Shape: {matrix.shape}")

    print(f"\n[6/6] Upserting to ChromaDB...")
    vs = get_vector_store()

    ids   = []
    metas = []

    for idx, row in df.iterrows():
        ids.append(str(idx))
        metas.append({
            "movie_id":     int(row.get("id", 0)),
            "title":        str(row["title"]),
            "overview":     str(row["overview"])[:500],
            "genres":       ", ".join(row["genre_list"]),
            "vote_average": float(row["vote_average"]),
            "vote_count":   int(row["vote_count"]),
            "release_year": str(row["release_year"]),
            "poster_url":   str(row["poster_url"]),
            "director":     str(row["director"]),
            "cast":         ", ".join(row["cast_list"]),
            "tagline":      str(row.get("tagline", "")),
        })

    UPSERT_BATCH = 500
    for i in range(0, len(ids), UPSERT_BATCH):
        vs.upsert(
            ids=ids[i: i + UPSERT_BATCH],
            embeddings=matrix[i: i + UPSERT_BATCH].tolist(),
            documents=documents[i: i + UPSERT_BATCH],
            metadatas=metas[i: i + UPSERT_BATCH],
        )
        print(f"    Upserted {min(i + UPSERT_BATCH, len(ids)):,}/{len(ids):,}", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"\n\nDone in {elapsed:.1f}s — {vs.count():,} docs indexed.")
    print("Run: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
