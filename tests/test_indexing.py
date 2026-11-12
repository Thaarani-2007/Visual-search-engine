"""
Unit and Integration Tests for Phase 4: FAISS Vector Indexing & Retrieval.
"""

from pathlib import Path
import faiss
import numpy as np
import pytest

from src.indexing.faiss_index import FAISSIndexManager


@pytest.fixture
def sample_embeddings():
    """Generate 50 normalized 512-D vectors for indexing tests."""
    np.random.seed(42)
    vectors = np.random.randn(50, 512).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def test_flat_index_exact_match(sample_embeddings):
    """Verify that querying an exact vector from the index returns similarity == 1.0."""
    manager = FAISSIndexManager()
    index = manager.build_flat_index(sample_embeddings)

    # Query with 5th vector
    query = sample_embeddings[5:6]
    scores, indices = index.search(query, k=5)

    assert indices[0, 0] == 5, f"Expected top match index 5, got {indices[0, 0]}"
    assert abs(scores[0, 0] - 1.0) < 1e-5, f"Expected similarity 1.0, got {scores[0, 0]}"


def test_hnsw_retrieval_and_score_bounds(sample_embeddings):
    """Verify HNSW retrieval, score ranges, and monotonic ranking."""
    manager = FAISSIndexManager()
    index = manager.build_hnsw_index(sample_embeddings, m=16, ef_construction=64, ef_search=32)

    query = sample_embeddings[12:13]
    scores, indices = index.search(query, k=5)

    # Exact vector must be top retrieved match
    assert indices[0, 0] == 12
    assert abs(scores[0, 0] - 1.0) < 1e-4

    # Monotonicity test: scores must be non-increasing
    for r in range(len(scores[0]) - 1):
        assert scores[0, r] >= scores[0, r + 1] - 1e-6, "Scores are not monotonically decreasing!"

    # Score bounds test: cosine similarity on unit vectors must be in [-1.0, 1.0]
    assert (scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all()


def test_index_persistence_roundtrip(tmp_path, sample_embeddings):
    """Verify FAISS index serialization and disk reloading."""
    manager = FAISSIndexManager()
    index_orig = manager.build_flat_index(sample_embeddings)

    save_path = tmp_path / "test_flat.index"
    faiss.write_index(index_orig, str(save_path))

    index_loaded = faiss.read_index(str(save_path))
    assert index_loaded.ntotal == index_orig.ntotal

    query = sample_embeddings[:3]
    scores_orig, ids_orig = index_orig.search(query, k=3)
    scores_load, ids_load = index_loaded.search(query, k=3)

    assert np.array_equal(ids_orig, ids_load)
    assert np.allclose(scores_orig, scores_load, atol=1e-6)

