# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# Install all Python deps into an isolated prefix so the final image
# only copies compiled wheels, not the full pip cache (~600 MB saved).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (e.g. tokenizers, chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a custom prefix — copied into the final stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# Clean slim image with only what's needed to run the API.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="CineMatch API"
LABEL org.opencontainers.image.description="Production-grade movie recommendation API"
LABEL org.opencontainers.image.version="1.0.0"

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user — never run as root in production
RUN useradd --create-home --shell /bin/bash cinematch \
 && mkdir -p /app/logs /app/.chromadb /app/data \
 && chown -R cinematch:cinematch /app

USER cinematch

# Copy application source
COPY --chown=cinematch:cinematch app/       ./app/
COPY --chown=cinematch:cinematch scripts/   ./scripts/
COPY --chown=cinematch:cinematch tests/     ./tests/

# Environment defaults (overridden by docker-compose or cloud env vars)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIR=/app/.chromadb \
    MOVIES_CSV=/app/data/tmdb_5000_movies.csv \
    CREDITS_CSV=/app/data/tmdb_5000_credits.csv

EXPOSE 8000

# Healthcheck — Docker and Render both use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use exec form — signals go directly to uvicorn, not a shell
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--access-log"]
