"""
Comprehensive FAISS Indexing and Retrieval Benchmarking Script.

Evaluates and compares:
1. FAISS IndexFlatIP (Exact Cosine Baseline) vs FAISS IndexHNSWFlat (Approximate).
2. Build latency and disk/RAM footprint.
3. Query search latency: Mean, p50, p95.
4. Relative Recall@1 and Recall@5 of HNSW against FlatIP exact ground truth.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import faiss
import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.indexing.extractor import OfflineEmbeddingExtractor
from src.indexing.faiss_index import FAISSIndexManager


def benchmark_vector_indices(config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    index_dir = PROJECT_ROOT / cfg["indexing"]["index_dir"]
    emb_file = index_dir / "embeddings.npy"

    print("=" * 70)
    print("FAISS VECTOR RETRIEVAL BENCHMARK")
    print("=" * 70)

    # 1. Acquire Embeddings
    if not emb_file.is_file():
        print("Extracting embeddings using trained model...")
        extractor = OfflineEmbeddingExtractor(config_path=config_path)
        embeddings, image_ids, metadata_lookup = extractor.extract_dataset_embeddings()
        extractor.save_extracted_artifacts(embeddings, image_ids, metadata_lookup)
    else:
        print(f"Loading existing embeddings from: {emb_file}")
        embeddings = np.load(emb_file)

    n_samples, dim = embeddings.shape
    print(f"Indexed Dataset Size: {n_samples} vectors | Dimension: {dim}-D")

    # 2. Build Exact Index (IndexFlatIP)
    manager = FAISSIndexManager(config_path=config_path)
    t0 = time.perf_counter()
    flat_index = manager.build_flat_index(embeddings)
    t_build_flat = time.perf_counter() - t0

    # 3. Build Approximate Index (IndexHNSWFlat)
    t0 = time.perf_counter()
    hnsw_index = manager.build_hnsw_index(embeddings)
    t_build_hnsw = time.perf_counter() - t0

    # Persist indices to disk
    manager.save_indices()

    # 4. Latency Benchmarking (Run 100 queries)
    n_queries = min(100, n_samples)
    np.random.seed(cfg["seed"])
    query_indices = np.random.choice(n_samples, size=n_queries, replace=False)
    query_vectors = embeddings[query_indices]  # [n_queries, 512]

    # Warmup
    _ = flat_index.search(query_vectors[:5], k=5)
    _ = hnsw_index.search(query_vectors[:5], k=5)

    # Measure FlatIP Latency
    flat_latencies = []
    flat_results = []
    for q in query_vectors:
        q_2d = np.expand_dims(q, axis=0)
        t_start = time.perf_counter()
        scores, ids = flat_index.search(q_2d, k=5)
        flat_latencies.append((time.perf_counter() - t_start) * 1000)  # ms
        flat_results.append(ids[0])

    # Measure HNSWFlat Latency
    hnsw_latencies = []
    hnsw_results = []
    for q in query_vectors:
        q_2d = np.expand_dims(q, axis=0)
        t_start = time.perf_counter()
        scores, ids = hnsw_index.search(q_2d, k=5)
        hnsw_latencies.append((time.perf_counter() - t_start) * 1000)  # ms
        hnsw_results.append(ids[0])

    # 5. Compute Recall@K of HNSW relative to Exact Flat Ground Truth
    recall_at_1_matches = 0
    recall_at_5_intersections = 0

    for i in range(n_queries):
        true_top5 = set(flat_results[i])
        hnsw_top5 = set(hnsw_results[i])

        # Recall@1: exact top item match
        if flat_results[i][0] == hnsw_results[i][0]:
            recall_at_1_matches += 1

        # Recall@5: overlap of top-5 sets
        overlap = len(true_top5.intersection(hnsw_top5))
        recall_at_5_intersections += (overlap / 5.0)

    recall_1 = (recall_at_1_matches / n_queries) * 100.0
    recall_5 = (recall_at_5_intersections / n_queries) * 100.0

    # Latency Percentiles
    flat_mean = np.mean(flat_latencies)
    flat_p50 = np.percentile(flat_latencies, 50)
    flat_p95 = np.percentile(flat_latencies, 95)

    hnsw_mean = np.mean(hnsw_latencies)
    hnsw_p50 = np.percentile(hnsw_latencies, 50)
    hnsw_p95 = np.percentile(hnsw_latencies, 95)

    # Disk Sizes
    flat_size_kb = (manager.flat_path.stat().st_size / 1024) if manager.flat_path.is_file() else 0
    hnsw_size_kb = (manager.hnsw_path.stat().st_size / 1024) if manager.hnsw_path.is_file() else 0

    print("\n" + "=" * 70)
    print("BENCHMARK REPORT: EXACT vs. APPROXIMATE RETRIEVAL")
    print("=" * 70)
    print(f"{'Metric':<32} | {'IndexFlatIP (Exact)':<18} | {'IndexHNSWFlat (Approx)':<18}")
    print("-" * 74)
    print(f"{'Index Build Time (ms)':<32} | {t_build_flat * 1000:<18.2f} | {t_build_hnsw * 1000:<18.2f}")
    print(f"{'Index File Size on Disk (KB)':<32} | {flat_size_kb:<18.1f} | {hnsw_size_kb:<18.1f}")
    print(f"{'Mean Search Latency (ms)':<32} | {flat_mean:<18.4f} | {hnsw_mean:<18.4f}")
    print(f"{'p50 Search Latency (ms)':<32} | {flat_p50:<18.4f} | {hnsw_p50:<18.4f}")
    print(f"{'p95 Search Latency (ms)':<32} | {flat_p95:<18.4f} | {hnsw_p95:<18.4f}")
    print(f"{'Recall@1 (vs Exact Truth)':<32} | {'100.0% (Baseline)':<18} | {f'{recall_1:.1f}%':<18}")
    print(f"{'Recall@5 (vs Exact Truth)':<32} | {'100.0% (Baseline)':<18} | {f'{recall_5:.1f}%':<18}")
    print("=" * 70 + "\n")

    return {
        "flat": {"build_ms": t_build_flat * 1000, "size_kb": flat_size_kb, "latency_mean": flat_mean},
        "hnsw": {"build_ms": t_build_hnsw * 1000, "size_kb": hnsw_size_kb, "latency_mean": hnsw_mean, "recall_5": recall_5},
    }


if __name__ == "__main__":
    benchmark_vector_indices()

