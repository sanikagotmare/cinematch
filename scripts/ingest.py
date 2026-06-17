"""
scripts/ingest.py
──────────────────
One-time data pipeline: CSV → text → embeddings → ChromaDB.
"""
from __future__ import annotations
import ast
import os
import sys
import time
from pathlib import Path

# ── Fix Python path FIRST before any other imports ────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Debug: print path so we can see it in Render logs
print(f"ROOT={ROOT}", flush=True)
print(f"sys.path={sys.path[:3]}", flush=True)
print(f"app exists: {(ROOT / 'app').exists()}", flush=True)
print(f"app/models exists: {(ROOT / 'app' / 'models').exists()}", flush=True)

import pandas as pd
import numpy as np

from app.core.config import get_settings
from app.db.vector_store import get_vector_store
from app.models.embedder import get_embedder

settings = get_settings()
BATCH_SIZE = 64


def _parse_names(value: str, key: str = "name", limit: int | None = None) -> list[str]:
    try:
        items = ast.literal_eval(value)
        names = [i[key] for i in items if isinstance(i, dict) and key in i]
        return names[:limit] if limit else names
    except Exception:
        return []


def _parse_director(crew_str: str) -> str:
    try:
        crew = ast.literal_eval(crew_str)
        return next(
            (c["name"] for c in crew if isinstance(c, dict) and c.get("job") == "Director"),
            "",
        )
    except Exception:
        return ""


def _build_document(row: pd.Series) -> str:
    genres   = " ".join(row["genre_list"])
    keywords = " ".join(row["keyword_list"][:15])
    cast     = " ".join(row["cast_list"][:5])
    director = row["director"]
    overview = row["overview"] or ""
    return (
        f"{overview} "
        f"{genres} {genres} {genres} "
        f"{keywords} {keywords} "
        f"{cast} "
        f"{director}"
    ).strip()


def main() -> None:
    t0 = time.perf_counter()
    print("─" * 60)
    print("  CineMatch Ingestion Pipeline")
    print("─" * 60)

    print(f"\n[1/5] Loading CSVs from '{settings.DATA_DIR}'…")
    movies_path  = Path(settings.MOVIES_CSV)
    credits_path = Path(settings.CREDITS_CSV)

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing: {movies_path}. Download from Kaggle.")
    if not credits_path.exists():
        raise FileNotFoundError(f"Missing: {credits_path}. Download from Kaggle.")

    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    print(f"    Loaded {len(movies):,} movies and {len(credits):,} credit records.")

    print("\n[2/5] Merging and parsing JSON columns…")
    df = movies.merge(credits[["title", "cast", "crew"]], on="title", how="left")

    df["genre_list"]    = df["genres"].fillna("[]").apply(_parse_names)
    df["keyword_list"]  = df["keywords"].fillna("[]").apply(_parse_names)
    df["cast_list"]     = df["cast"].fillna("[]").apply(lambda x: _parse_names(x, limit=5))
    df["director"]      = df["crew"].fillna("[]").apply(_parse_director)
    df["overview"]      = df["overview"].fillna("")
    df["vote_average"]  = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"]    = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["release_year"]  = df["release_date"].astype(str).str[:4]

    df = df[
        (df["title"].notna()) &
        (df["vote_count"] >= 20) &
        (df["overview"].str.len() > 20)
    ].reset_index(drop=True)

    # ── Fix duplicate IDs — use row index to make every ID unique ──────────────
    df["unique_id"] = df.index.astype(str)
    print(f"    {len(df):,} movies after quality filter.")

    print("\n[3/5] Building combined text documents…")
    df["document"] = df.apply(_build_document, axis=1)

    print(f"\n[4/5] Generating embeddings (batch_size={BATCH_SIZE})…")
    embedder  = get_embedder()
    documents = df["document"].tolist()

    all_embeddings: list[np.ndarray] = []
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(documents), BATCH_SIZE):
        batch     = documents[i : i + BATCH_SIZE]
        embs      = embedder.encode(batch, normalize=True)
        all_embeddings.append(embs)
        batch_num = i // BATCH_SIZE + 1
        print(f"    Batch {batch_num}/{total_batches} — {len(batch)} movies", end="\r")

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"\n    Done. Embedding matrix: {embeddings_matrix.shape}")

    print(f"\n[5/5] Upserting to ChromaDB at '{settings.CHROMA_PERSIST_DIR}'…")
    vector_store = get_vector_store()

    ids: list[str]       = []
    metadatas: list[dict] = []

    for idx, row in df.iterrows():
        # Use row index as ID — guaranteed unique even if movie IDs repeat
        ids.append(str(idx))
        metadatas.append({
            "movie_id":     int(row.get("id", 0)),
            "title":        str(row["title"]),
            "overview":     str(row["overview"])[:500],
            "genres":       ", ".join(row["genre_list"]),
            "vote_average": float(row["vote_average"]),
            "release_year": str(row["release_year"]),
            "poster_path":  str(row.get("poster_path") or ""),
            "director":     str(row["director"]),
            "cast":         ", ".join(row["cast_list"]),
        })

    UPSERT_BATCH = 500
    for i in range(0, len(ids), UPSERT_BATCH):
        vector_store.upsert(
            ids=ids[i : i + UPSERT_BATCH],
            embeddings=embeddings_matrix[i : i + UPSERT_BATCH].tolist(),
            documents=documents[i : i + UPSERT_BATCH],
            metadatas=metadatas[i : i + UPSERT_BATCH],
        )
        print(f"    Upserted {min(i + UPSERT_BATCH, len(ids)):,}/{len(ids):,}", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"\n\n✓ Ingestion complete in {elapsed:.1f}s")
    print(f"  {vector_store.count():,} documents now indexed in ChromaDB.")
    print("─" * 60)
    print("  You can now start the API: uvicorn app.main:app --reload")
    print("─" * 60)


if __name__ == "__main__":
    main()
