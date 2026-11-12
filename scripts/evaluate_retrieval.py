"""
Quantitative Retrieval Evaluation Script.

Compares:
1. ImageNet-Pretrained ResNet-50 Baseline (Before Metric Learning)
2. Fine-Tuned ResNet-50 with Online Semi-Hard Triplet Margin Loss (After Metric Learning)

Computes:
- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
on the held-out identity-disjoint test set.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.models.siamese_net import ResNet50MetricNet
from src.utils.metrics import compute_retrieval_metrics


def extract_features_for_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, pids, _, _ in loader:
            images = images.to(device)
            embs = model(images).cpu().numpy()
            embeddings.append(embs)
            labels.extend([p.item() if hasattr(p, "item") else p for p in pids])
    return np.vstack(embeddings).astype(np.float32), np.array(labels)


def evaluate_model_retrieval(config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    metadata_file = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    ckpt_path = PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / cfg["training"]["best_model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHASE 6: RETRIEVAL EVALUATION ON IDENTITY-DISJOINT TEST SET")
    print("=" * 70)

    # 1. Load Test Dataset
    test_dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split="test",
        is_triplet=False,
        image_size=cfg["preprocessing"]["image_size"],
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Loaded Test Split: {len(test_dataset)} images across {test_dataset.df['product_id'].nunique()} unseen identities.")

    # 2. Baseline Model (ImageNet Pretrained ResNet-50)
    print("\n[1/2] Evaluating ImageNet-Pretrained Baseline (Before Metric Learning)...")
    baseline_model = ResNet50MetricNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=True,
        freeze_early_layers=False,
    ).to(device)
    base_embs, base_labels = extract_features_for_eval(baseline_model, test_loader, device)

    # Build Exact Flat index for baseline
    base_index = faiss.IndexFlatIP(cfg["model"]["embedding_dim"])
    base_index.add(base_embs)
    # Search for top-11 (item 0 will be self, so we look at items 1..10)
    _, base_retrieved = base_index.search(base_embs, k=11)
    base_retrieved_clean = base_retrieved[:, 1:]  # Exclude self-match at rank 0
    base_metrics = compute_retrieval_metrics(base_labels, base_labels, base_retrieved_clean, k_list=[1, 5, 10])

    # 3. Fine-Tuned Metric Learning Model
    print("\n[2/2] Evaluating Fine-Tuned ResNet-50 Metric Model (After Metric Learning)...")
    trained_model = ResNet50MetricNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=False,
        freeze_early_layers=False,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    trained_model.to(device)
    trained_embs, trained_labels = extract_features_for_eval(trained_model, test_loader, device)

    # Build Exact Flat index for trained model
    trained_index = faiss.IndexFlatIP(cfg["model"]["embedding_dim"])
    trained_index.add(trained_embs)
    _, trained_retrieved = trained_index.search(trained_embs, k=11)
    trained_retrieved_clean = trained_retrieved[:, 1:]  # Exclude self-match at rank 0
    trained_metrics = compute_retrieval_metrics(trained_labels, trained_labels, trained_retrieved_clean, k_list=[1, 5, 10])

    # 4. Display Comparison Report
    print("\n" + "=" * 70)
    print("RETRIEVAL ACCURACY REPORT: BASELINE vs. FINE-TUNED METRIC LEARNING")
    print("=" * 70)
    print(f"{'Metric':<20} | {'ImageNet Baseline':<20} | {'Fine-Tuned ResNet-50':<20} | {'Delta (Gain)':<12}")
    print("-" * 74)
    for k in ["Recall@1", "Recall@5", "Recall@10", "MRR"]:
        b_val = base_metrics[k]
        t_val = trained_metrics[k]
        diff = t_val - b_val
        delta_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        print(f"{k:<20} | {b_val:<20} | {t_val:<20} | {delta_str:<12}")
    print("=" * 70 + "\n")

    return {"baseline": base_metrics, "fine_tuned": trained_metrics}


if __name__ == "__main__":
    evaluate_model_retrieval()
