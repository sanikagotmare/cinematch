"""
tests/test_recommend.py
────────────────────────
Async integration tests using httpx.AsyncClient as the test transport.
These test the full request/response cycle including Pydantic validation.

Run with:
    pytest tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store_docs" in data
    assert "cache_stats" in data


@pytest.mark.asyncio
async def test_recommend_returns_valid_schema():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/recommend/Inception")
    assert response.status_code == 200
    data = response.json()
    assert data["query_title"] == "Inception"
    assert isinstance(data["recommendations"], list)
    if data["recommendations"]:
        first = data["recommendations"][0]
        assert "title" in first
        assert 0.0 <= first["similarity_score"] <= 1.0
        assert 0.0 <= first["vote_average"] <= 10.0


@pytest.mark.asyncio
async def test_recommend_cache_hit():
    """Second request for same title must return cached=True."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/api/v1/recommend/Inception")          # warm cache
        response = await client.get("/api/v1/recommend/Inception")  # should hit
    data = response.json()
    assert data["cached"] is True


@pytest.mark.asyncio
async def test_recommend_top_k_respected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/recommend/Inception?top_k=3")
    data = response.json()
    assert len(data["recommendations"]) <= 3


@pytest.mark.asyncio
async def test_search_semantic():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/search?q=space+horror")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "space horror"
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_search_relevance_score_range():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/search?q=romantic+comedy+paris")
    for result in response.json()["results"]:
        assert 0.0 <= result["relevance_score"] <= 1.0


@pytest.mark.asyncio
async def test_recommend_invalid_top_k():
    """top_k=0 should return 422 Unprocessable Entity."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/recommend/Inception?top_k=0")
    assert response.status_code == 422
