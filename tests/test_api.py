"""
Integration Tests for FastAPI Visual Search Backend.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create test client using FastAPI Lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health reports healthy status and loaded components."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["index_loaded"] is True
    assert data["total_indexed_vectors"] > 0


def test_index_info_endpoint(client):
    """Verify /index-info returns vector dimension and index properties."""
    response = client.get("/index-info")
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == 512
    assert "HNSW" in data["index_type"]
    assert data["total_vectors"] > 0


def test_search_endpoint_valid_image(client):
    """Verify POST /search returns top-5 matches and latency metrics."""
    # Create synthetic test image in memory
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/search?top_k=5",
        files={"file": ("query.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["top_k"] == 5
    assert len(data["results"]) == 5

    # Verify result item schema
    for item in data["results"]:
        assert "rank" in item
        assert "similarity_score" in item
        assert "product_id" in item
        assert "image_url" in item
        # Score must be bounded in [-1.0, 1.0]
        assert -1.0 <= item["similarity_score"] <= 1.0

    # Verify latency breakdown
    lat = data["latency"]
    assert "preprocess_ms" in lat
    assert "inference_ms" in lat
    assert "search_ms" in lat
    assert "total_ms" in lat
    assert lat["total_ms"] > 0


def test_search_endpoint_invalid_file_type(client):
    """Verify POST /search rejects non-image uploads with HTTP 400."""
    response = client.post(
        "/search",
        files={"file": ("text.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported media type" in response.json()["detail"]


def test_search_endpoint_corrupted_image(client):
    """Verify POST /search handles corrupted bytes gracefully with HTTP 400."""
    response = client.post(
        "/search",
        files={"file": ("broken.jpg", b"\xFF\xD8\xFF\xE0corrupted_data_stream", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupted or invalid image" in response.json()["detail"]

