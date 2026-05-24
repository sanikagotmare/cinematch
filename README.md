# CineMatch 🎬

> **Production-grade movie recommendation API** built with FastAPI, sentence-transformer embeddings, and ChromaDB vector search.

[![CI](https://github.com/YOUR_USERNAME/cinematch/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/cinematch/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What it does

CineMatch recommends movies using **dense vector embeddings** — not keyword matching or TF-IDF. Searching *"space horror survival"* returns Alien, Event Horizon, and Gravity because the system understands semantic meaning, not just shared words.

**Live demo:** `https://cinematch.onrender.com` *(spins up in ~30s on free tier)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (async)                       │
│                                                              │
│   GET /api/v1/recommend/{title}   GET /api/v1/search?q=     │
│              │                              │                │
│              ▼                              ▼                │
│      ┌───────────────┐            ┌──────────────────┐      │
│      │  TTL Cache    │            │  Semantic Search  │      │
│      │  (in-memory)  │            │  Service          │      │
│      └──────┬────────┘            └────────┬─────────┘      │
│             │ miss                          │                │
│             ▼                              ▼                │
│      ┌──────────────────────────────────────────────┐       │
│      │         all-MiniLM-L6-v2 Embedder            │       │
│      │    (sentence-transformers · 384 dim · CPU)   │       │
│      └──────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│      ┌──────────────────────────────────────────────┐       │
│      │      ChromaDB  (persistent HNSW index)       │       │
│      │      4,800 movies · cosine similarity        │       │
│      └──────────────────┬───────────────────────────┘       │
│                         │  top-k neighbours                  │
│                         ▼                                    │
│      ┌──────────────────────────────────────────────┐       │
│      │  TMDB Watch Providers  (async httpx)         │       │
│      │  asyncio.gather() — all calls in parallel    │       │
│      └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI 0.111 | Async-native, auto Swagger docs, Pydantic validation |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, ~80 MB, state-of-the-art on CPU |
| Vector DB | ChromaDB (HNSW) | Persistent, no separate server, cosine similarity |
| HTTP client | httpx (async) | Connection pooling, TMDB watch providers |
| Caching | In-memory TTL cache | Redis-compatible interface, swap-ready |
| Containerisation | Docker + docker-compose | One-command local setup |
| CI/CD | GitHub Actions | Tests on every push, Docker build check |
| Deployment | Render.com | Free tier, auto-deploy on push |

---

## Quickstart

### Option A — Docker (recommended, one command)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch

# 2. Add your Kaggle CSVs
mkdir data
cp ~/Downloads/tmdb_5000_movies.csv  data/
cp ~/Downloads/tmdb_5000_credits.csv data/

# 3. Copy env and add your TMDB API key (optional — works without it)
cp .env.example .env

# 4. Build + ingest + serve (all in one)
docker compose up --build

# API is live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Option B — Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up env
cp .env.example .env   # optionally add TMDB_API_KEY

# 3. Ingest data (one-time, ~3-8 min on CPU)
python scripts/ingest.py

# 4. Run the API
uvicorn app.main:app --reload

# 5. Run tests
pytest tests/ -v --asyncio-mode=auto
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/recommend/{title}` | Top-k similar movies by title |
| `GET` | `/api/v1/search?q=` | Semantic free-text search |
| `GET` | `/health` | Service health + cache stats |
| `GET` | `/docs` | Swagger interactive docs |

### Example requests

```bash
# Recommend similar movies
curl "http://localhost:8000/api/v1/recommend/Inception?top_k=5"

# Semantic search — no exact keywords needed
curl "http://localhost:8000/api/v1/search?q=space+horror+survival"
curl "http://localhost:8000/api/v1/search?q=romantic+comedy+paris"
curl "http://localhost:8000/api/v1/search?q=dystopian+rebellion"

# Health check
curl "http://localhost:8000/health"
```

### Example response — `/recommend/Inception`

```json
{
  "query_title": "Inception",
  "top_k": 5,
  "cached": false,
  "recommendations": [
    {
      "movie_id": 157336,
      "title": "Interstellar",
      "overview": "A team of explorers travel through a wormhole...",
      "genres": ["Adventure", "Drama", "Science Fiction"],
      "vote_average": 8.6,
      "release_year": "2014",
      "similarity_score": 0.8912,
      "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
      "streaming_providers": [
        { "provider_name": "Netflix", "display_priority": 1 }
      ]
    }
  ]
}
```

---

## Design decisions

### Why sentence-transformers over TF-IDF?

TF-IDF is a bag-of-words model — "thrilling space film" scores zero similarity against a movie described as "a cosmic, terrifying ordeal" because no tokens match. `all-MiniLM-L6-v2` encodes semantic meaning into a 384-dimensional vector space, so synonymous descriptions land close together geometrically. The tradeoff is inference time (~5 ms/query on CPU vs. ~0.1 ms for TF-IDF), which the cache absorbs on repeat queries.

### Why ChromaDB over in-RAM cosine matrix?

An in-RAM similarity matrix for 4,800 movies × 384 dimensions = ~7 MB of floats, which fits in memory — but computing it on every request is O(n) and doesn't scale. ChromaDB uses an HNSW (Hierarchical Navigable Small World) index: approximate nearest-neighbour search in O(log n), and the index persists to disk so ingestion is a one-time cost rather than a startup cost on every deploy.

### How is the cold-start problem addressed?

At **ingestion time**: all 4,800 embeddings are pre-computed and written to ChromaDB's on-disk HNSW index. At **startup time**: the embedding model and ChromaDB connection are both warmed during the FastAPI lifespan hook — before the first request is served. At **query time**: for known titles we skip the transformer entirely (encode the stored document, not the query string). For repeat queries, the TTL cache returns in < 1 ms.

### Why `asyncio.gather()` for streaming providers?

Fetching watch providers for 5 recommended movies sequentially would take `sum(5 × ~100 ms) = ~500 ms`. `asyncio.gather()` fires all 5 HTTP calls concurrently — total latency becomes `max(5 × ~100 ms) ≈ 100 ms`. This is the core benefit of the async model: I/O-bound tasks run in parallel without threads.

---

## Project structure

```
cinematch/
├── app/
│   ├── main.py              # FastAPI factory + lifespan hooks
│   ├── api/v1/
│   │   ├── recommend.py     # GET /recommend/{movie_title}
│   │   └── search.py        # GET /search?q=
│   ├── core/
│   │   ├── config.py        # Pydantic-settings env config
│   │   └── cache.py         # Generic TTL cache (Redis-compatible API)
│   ├── db/
│   │   └── vector_store.py  # ChromaDB client wrapper
│   ├── models/
│   │   └── embedder.py      # SentenceTransformer singleton
│   ├── schemas/
│   │   └── movie.py         # Pydantic request/response schemas
│   └── services/
│       ├── recommendation.py  # Orchestration: cache → embed → search → enrich
│       ├── semantic_search.py # Free-text semantic search
│       └── streaming.py       # Async TMDB watch provider fetcher
├── scripts/
│   └── ingest.py            # One-time: CSV → embeddings → ChromaDB
├── tests/
│   └── test_recommend.py    # Async integration tests
├── data/                    # ← put Kaggle CSVs here (gitignored)
├── Dockerfile               # Multi-stage production image
├── docker-compose.yml       # Local stack (ingest + api)
├── render.yaml              # One-click Render.com deployment
└── .github/workflows/ci.yml # GitHub Actions CI
```

---

## Deployment (Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render reads `render.yaml` automatically
4. In the **Environment** tab, add `TMDB_API_KEY` as a secret env var
5. Click **Deploy** — the build runs `ingest.py` then starts the server

> **Note:** The free tier uses ephemeral storage, so ChromaDB re-ingests on every deploy (~5 min). Upgrade to Starter ($7/mo) and add a **Disk** to persist the index between deploys.

---

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.

Download both files and place them in the `data/` directory:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The data files are gitignored (too large for GitHub). The ingestion script parses JSON columns, builds rich combined-text documents, and generates 384-dim embeddings stored in ChromaDB.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [YOUR NAME] · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Portfolio](https://yoursite.com)*
