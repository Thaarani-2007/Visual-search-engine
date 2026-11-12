"""
FAISS Vector Index Management and Search Engine.

Features:
- Exact search baseline via FAISS IndexFlatIP.
- Approximate Nearest Neighbor (ANN) search via FAISS IndexHNSWFlat.
- Configurable HNSW parameters (M, efConstruction, efSearch).
- Persistent index serialization & disk loading.
- High-level query interface returning similarity scores, IDs, and image paths.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import faiss
import numpy as np

from src.config import PROJECT_ROOT, load_config


class FAISSIndexManager:
    """Manages construction, persistence, and querying of FAISS vector indices."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.dimension = self.cfg["model"]["embedding_dim"]
        self.index_dir = PROJECT_ROOT / self.cfg["indexing"]["index_dir"]
        self.flat_path = PROJECT_ROOT / self.cfg["indexing"]["faiss_exact_file"]
        self.hnsw_path = PROJECT_ROOT / self.cfg["indexing"]["faiss_hnsw_file"]
        self.meta_path = PROJECT_ROOT / self.cfg["indexing"]["metadata_lookup_file"]

        self.flat_index: Optional[faiss.IndexFlatIP] = None
        self.hnsw_index: Optional[faiss.IndexHNSWFlat] = None
        self.metadata_lookup: Dict[int, dict] = {}

    def build_flat_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Construct exact-search baseline index using Inner Product.
        Since embeddings are L2-normalized, inner product equals exact cosine similarity.
        """
        assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings."
        assert embeddings.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"

        n_samples = embeddings.shape[0]
        start_time = time.perf_counter()

        # FAISS IndexFlatIP: Exact brute-force inner product scan
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        elapsed = time.perf_counter() - start_time
        print(f"Built IndexFlatIP (Exact) with {index.ntotal} vectors in {elapsed * 1000:.2f} ms.")
        self.flat_index = index
        return index

    def build_hnsw_index(
        self,
        embeddings: np.ndarray,
        m: int = None,
        ef_construction: int = None,
        ef_search: int = None,
    ) -> faiss.IndexHNSWFlat:
        """
        Construct Hierarchical Navigable Small World (HNSW) graph index.
        Parameters:
            m: number of connections per node (default: 32).
            ef_construction: candidate list size during graph build (default: 200).
            ef_search: candidate list size during query traversal (default: 64).
        """
        assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings."
        if m is None:
            m = self.cfg["indexing"].get("hnsw_m", 32)
        if ef_construction is None:
            ef_construction = self.cfg["indexing"].get("hnsw_ef_construction", 200)
        if ef_search is None:
            ef_search = self.cfg["indexing"].get("hnsw_ef_search", 64)

        start_time = time.perf_counter()

        # Initialize HNSW with METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        index.add(embeddings)
        elapsed = time.perf_counter() - start_time

        print(
            f"Built IndexHNSWFlat (M={m}, efC={ef_construction}, efS={ef_search}) "
            f"with {index.ntotal} vectors in {elapsed * 1000:.2f} ms."
        )
        self.hnsw_index = index
        return index

    def set_ef_search(self, ef_search: int) -> None:
        """Dynamically tune query-time search depth without rebuilding index."""
        if self.hnsw_index is not None:
            self.hnsw_index.hnsw.efSearch = ef_search

    def save_indices(self) -> None:
        """Persist indices to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.flat_index is not None:
            faiss.write_index(self.flat_index, str(self.flat_path))
            print(f"Persisted exact index to: {self.flat_path} ({self.flat_path.stat().st_size / 1024:.1f} KB)")
        if self.hnsw_index is not None:
            faiss.write_index(self.hnsw_index, str(self.hnsw_path))
            print(f"Persisted HNSW index to:  {self.hnsw_path} ({self.hnsw_path.stat().st_size / 1024:.1f} KB)")

    def load_indices(self, load_flat: bool = True, load_hnsw: bool = True) -> None:
        """Load persistent indices from disk into memory."""
        if load_flat and self.flat_path.is_file():
            self.flat_index = faiss.read_index(str(self.flat_path))
            print(f"Loaded IndexFlatIP ({self.flat_index.ntotal} vectors) from {self.flat_path}")
        if load_hnsw and self.hnsw_path.is_file():
            self.hnsw_index = faiss.read_index(str(self.hnsw_path))
            ef_search = self.cfg["indexing"].get("hnsw_ef_search", 64)
            self.hnsw_index.hnsw.efSearch = ef_search
            print(f"Loaded IndexHNSWFlat ({self.hnsw_index.ntotal} vectors, efSearch={ef_search}) from {self.hnsw_path}")

        # Load metadata lookup
        if self.meta_path.is_file():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                # JSON keys are strings, convert back to int
                data = json.load(f)
                self.metadata_lookup = {int(k): v for k, v in data.items()}
            print(f"Loaded metadata lookup ({len(self.metadata_lookup)} entries) from {self.meta_path}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        use_approximate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute nearest-neighbor retrieval for a 512-D normalized query vector.
        Args:
            query_embedding: [1, 512] or [512] normalized vector.
            top_k: number of top visual matches to retrieve.
            use_approximate: whether to use HNSW (True) or FlatIP (False).
        Returns:
            List of dicts: [ {rank, similarity_score, product_id, image_id, relative_path}, ... ]
        """
        # Ensure 2D float32 array: [1, 512]
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        # Normalize query vector if needed
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        if norm[0, 0] > 0 and abs(norm[0, 0] - 1.0) > 1e-4:
            query = query / norm

        index = self.hnsw_index if (use_approximate and self.hnsw_index is not None) else self.flat_index
        if index is None:
            raise RuntimeError("No FAISS index is loaded in memory. Call load_indices() first.")

        # FAISS search: returns distances (Inner Products) and nearest neighbor indices
        scores, indices = index.search(query, top_k)

        results = []
        for rank in range(top_k):
            faiss_id = indices[0, rank]
            if faiss_id == -1:
                continue  # Fewer items than top_k

            similarity_score = float(scores[0, rank])
            meta = self.metadata_lookup.get(faiss_id, {})

            results.append({
                "rank": rank + 1,
                "similarity_score": round(similarity_score, 5),
                "product_id": meta.get("product_id"),
                "image_id": meta.get("image_id"),
                "relative_path": meta.get("relative_path"),
            })

        return results

