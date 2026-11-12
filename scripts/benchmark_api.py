"""
API Retrieval Latency and Concurrency Benchmarking Script.

Measures:
- Preprocessing latency (ms)
- ResNet-50 inference latency (ms)
- FAISS HNSW search latency (ms)
- Total endpoint latency: mean, p50, p95, p99 (ms)
- Throughput (Requests Per Second - RPS) under concurrent load.
- Validates the project performance SLA: < 200 ms end-to-end retrieval.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import app


def benchmark_endpoint_performance(n_requests: int = 50, concurrency: int = 4) -> dict:
    print("=" * 70)
    print("FASTAPI RETRIEVAL ENDPOINT LATENCY & THROUGHPUT BENCHMARK")
    print(f"Total Requests: {n_requests} | Concurrency Level: {concurrency} threads")
    print("=" * 70)

    # Generate synthetic JPEG query bytes
    img = Image.new("RGB", (224, 224), color=(80, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    query_bytes = buf.getvalue()

    total_latencies = []
    inference_latencies = []
    search_latencies = []
    preprocess_latencies = []

    with TestClient(app) as client:
        # Warmup
        _ = client.post("/search?top_k=5", files={"file": ("warmup.jpg", query_bytes, "image/jpeg")})

        def send_request():
            t0 = time.perf_counter()
            resp = client.post(
                "/search?top_k=5",
                files={"file": ("query.jpg", query_bytes, "image/jpeg")},
            )
            wall_time = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                lat = data["latency"]
                return {
                    "wall_ms": wall_time,
                    "pre_ms": lat["preprocess_ms"],
                    "inf_ms": lat["inference_ms"],
                    "search_ms": lat["search_ms"],
                }
            return None

        # Execute concurrent requests
        t_start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(n_requests)]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    total_latencies.append(res["wall_ms"])
                    preprocess_latencies.append(res["pre_ms"])
                    inference_latencies.append(res["inf_ms"])
                    search_latencies.append(res["search_ms"])

        total_wall_time = time.perf_counter() - t_start_all

    throughput = len(total_latencies) / total_wall_time

    # Compute percentiles
    p50 = np.percentile(total_latencies, 50)
    p95 = np.percentile(total_latencies, 95)
    p99 = np.percentile(total_latencies, 99)
    mean_inf = np.mean(inference_latencies)
    mean_search = np.mean(search_latencies)
    mean_pre = np.mean(preprocess_latencies)
    mean_total = np.mean(total_latencies)

    print("\n" + "=" * 70)
    print("API PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"{'Performance Metric':<35} | {'Measured Value':<25}")
    print("-" * 65)
    print(f"{'Throughput (Requests / Second)':<35} | {throughput:<25.2f} req/s")
    print(f"{'Mean Preprocessing Latency':<35} | {mean_pre:<25.2f} ms")
    print(f"{'Mean ResNet-50 Inference Latency':<35} | {mean_inf:<25.2f} ms")
    print(f"{'Mean FAISS Search Latency':<35} | {mean_search:<25.3f} ms")
    print(f"{'Mean End-to-End Latency':<35} | {mean_total:<25.2f} ms")
    print(f"{'p50 End-to-End Latency (Median)':<35} | {p50:<25.2f} ms")
    print(f"{'p95 End-to-End Latency':<35} | {p95:<25.2f} ms")
    print(f"{'p99 End-to-End Latency':<35} | {p99:<25.2f} ms")
    print("-" * 65)
    if p95 < 200.0:
        print(f"✓ SLA STATUS: PASSED (p95 latency {p95:.1f} ms is strictly within < 200 ms target)")
    else:
        print(f"! SLA STATUS: EXCEEDED (p95: {p95:.1f} ms)")
    print("=" * 70 + "\n")

    return {
        "throughput_rps": throughput,
        "mean_latency": mean_total,
        "p50_latency": p50,
        "p95_latency": p95,
        "p99_latency": p99,
    }


if __name__ == "__main__":
    benchmark_endpoint_performance(n_requests=30, concurrency=2)

