"""
scripts/ingest.py - Standalone version for Render deployment
All app code is inlined here to avoid import issues.
"""
import ast
import os
import sys
import time
import requests
from pathlib import Path

# ── Ensure we run from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

print(f"Working dir: {os.getcwd()}", flush=True)
print(f"ROOT: {ROOT}", flush=True)
print(f"app/ exists: {(ROOT / 'app').exists()}", flush=True)
print(f"Python: {sys.version}", flush=True)

import pandas as pd
import numpy as np

# ── Settings (inline to avoid import issues) ───────────────────────────────────
MOVIES_CSV  = os.environ.get("MOVIES_CSV",  str(ROOT / "data" / "tmdb_5000_movies.csv"))
CREDITS_CSV = os.environ.get("CREDITS_CSV", str(ROOT / "data" / "tmdb_5000_credits.csv"))
CHROMA_DIR  = os.environ.get("CHROMA_PERSIST_DIR", str(ROOT / ".chromadb"))
TMDB_KEY    = os.environ.get("TMDB_API_KEY", "")
COLLECTION  = "movies"
BATCH_SIZE  = 64

print(f"MOVIES_CSV: {MOVIES_CSV}", flush=True)
print(f"CREDITS_CSV: {CREDITS_CSV}", flush=True)
print(f"CHROMA_DIR: {CHROMA_DIR}", flush=True)

# ── Now safe to import app modules ─────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    print("All imports successful!", flush=True)
except ImportError as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_names(value, key="name", limit=None):
    try:
        items = ast.literal_eval(value)
        names = [i[key] for i in items if isinstance(i, dict) and key in i]
        return names[:limit] if limit else names
    except Exception:
        return []

def parse_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        return next(
            (c["name"] for c in crew if isinstance(c, dict) and c.get("job") == "Director"),
            "",
        )
    except Exception:
        return ""

def build_document(row):
    genres   = " ".join(row["genre_list"])
    keywords = " ".join(row["keyword_list"][:15])
    cast     = " ".join(row["cast_list"][:5])
    overview = row["overview"] or ""
    return f"{overview} {genres} {genres} {genres} {keywords} {keywords} {cast} {row['director']}".strip()

def fetch_poster(movie_id, api_key):
    if not api_key or api_key == "YOUR_TMDB_API_KEY_HERE":
        return ""
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.perf_counter()
    print("-" * 60, flush=True)
    print("  CineMatch Ingestion Pipeline", flush=True)
    print("-" * 60, flush=True)

    # 1. Load CSVs
    print(f"\n[1/5] Loading CSVs...", flush=True)
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    print(f"    Loaded {len(movies):,} movies and {len(credits):,} credits.", flush=True)

    # 2. Merge and parse
    print("\n[2/5] Merging and parsing...", flush=True)
    df = movies.merge(credits[["title", "cast", "crew"]], on="title", how="left")
    df["genre_list"]   = df["genres"].fillna("[]").apply(parse_names)
    df["keyword_list"] = df["keywords"].fillna("[]").apply(parse_names)
    df["cast_list"]    = df["cast"].fillna("[]").apply(lambda x: parse_names(x, limit=5))
    df["director"]     = df["crew"].fillna("[]").apply(parse_director)
    df["overview"]     = df["overview"].fillna("")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["release_year"] = df["release_date"].astype(str).str[:4]
    df = df[
        (df["title"].notna()) &
        (df["vote_count"] >= 20) &
        (df["overview"].str.len() > 20)
    ].reset_index(drop=True)
    print(f"    {len(df):,} movies after filter.", flush=True)

    # 3. Build documents
    print("\n[3/5] Building documents...", flush=True)
    df["document"] = df.apply(build_document, axis=1)

    # 4. Fetch posters
    print("\n[4/5] Fetching posters from TMDB...", flush=True)
    poster_urls = []
    for i, row in df.iterrows():
        url = fetch_poster(int(row.get("id", 0)), TMDB_KEY)
        poster_urls.append(url)
        if i % 200 == 0:
            print(f"    Posters: {i}/{len(df)}...", end="\r", flush=True)
        time.sleep(0.1)  # rate limit
    df["poster_url"] = poster_urls
    filled = sum(1 for u in poster_urls if u)
    print(f"\n    {filled:,}/{len(df):,} posters fetched.", flush=True)

    # 5. Generate embeddings
    print(f"\n[5/6] Generating embeddings...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = df["document"].tolist()
    all_embs  = []
    total     = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i: i + BATCH_SIZE]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
        print(f"    Batch {i//BATCH_SIZE+1}/{total}", end="\r", flush=True)
    matrix = np.vstack(all_embs)
    print(f"\n    Shape: {matrix.shape}", flush=True)

    # 6. Upsert to ChromaDB
    print(f"\n[6/6] Upserting to ChromaDB at {CHROMA_DIR}...", flush=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

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
            "poster_url":   str(df.at[idx, "poster_url"]),
            "director":     str(row["director"]),
            "cast":         ", ".join(row["cast_list"]),
        })

    UPSERT = 500
    for i in range(0, len(ids), UPSERT):
        col.upsert(
            ids=ids[i: i+UPSERT],
            embeddings=matrix[i: i+UPSERT].tolist(),
            documents=documents[i: i+UPSERT],
            metadatas=metas[i: i+UPSERT],
        )
        print(f"    Upserted {min(i+UPSERT, len(ids)):,}/{len(ids):,}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n\nDone in {elapsed:.1f}s — {col.count():,} docs indexed.", flush=True)


if __name__ == "__main__":
    main()
