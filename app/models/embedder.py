"""
app/models/embedder.py
───────────────────────
Singleton wrapper around the SentenceTransformer model.

Engineering decisions worth discussing:
  1. Singleton via module-level instance — the ~80 MB model is loaded
     once at application startup (inside the FastAPI lifespan) and
     reused across all requests. Loading it per-request would add
     2-4 seconds of cold latency every time.

  2. encode() is CPU-bound and blocking. We expose a run_in_executor
     helper so async route handlers can offload it to a thread pool
     without blocking the event loop — critical for keeping p99
     latency low under concurrent traffic.

  3. Normalisation — embeddings are L2-normalised before storage so
     that a dot-product in ChromaDB equals cosine similarity, which
     is both mathematically correct and faster.

Interview talking point — Cold Start:
  "The model is loaded during the ASGI lifespan 'startup' hook, not
   lazily on first request. This front-loads the one-time cost to
   deployment time rather than penalising the first real user."
"""
import asyncio
from functools import lru_cache
from typing import Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class Embedder:
    """Thin, async-friendly wrapper around SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready.")

    # ── Sync encode (used during ingestion, runs in a thread) ────────────────

    def encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Return L2-normalised embeddings of shape (N, 384).
        Normalisation turns dot-product into cosine similarity —
        dot(a, b) = cos_sim(a, b) when ||a|| = ||b|| = 1.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings  # shape: (len(texts), 384)

    # ── Async encode (used in request handlers) ───────────────────────────────

    async def async_encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Offload the CPU-bound encode() call to a thread pool so the
        async event loop stays free to handle other requests.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,                         # default ThreadPoolExecutor
            lambda: self.encode(texts, normalize),
        )


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Module-level singleton — safe to call from anywhere.
    First call loads the model; every subsequent call returns
    the already-loaded instance in O(1).
    """
    settings = get_settings()
    return Embedder(settings.EMBEDDING_MODEL)
