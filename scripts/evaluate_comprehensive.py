"""
Comprehensive Metric-Learning Retrieval Evaluation and Visualization Script.

Computes:
1. HitRate@1 (Recall@1): Is at least one relevant image ranked first?
2. HitRate@5 (Recall@5): Is at least one relevant image in the top 5?
3. Recall@10: Is at least one relevant image in the top 10?
4. Precision@5: What proportion of the top 5 results are relevant?
5. mAP@5: How well are relevant images ranked in the top 5?

Evaluates both:
- ImageNet-Pretrained Baseline (Before Metric Learning)
- Fine-Tuned ResNet-50 MetricNet (After Metric Learning)

Generates:
- Console report with exact metrics
- Visual retrieval comparison grid: visualizations/retrieval_visual_matches.png
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.models.siamese_net import ResNet50MetricNet
from src.utils.metrics import compute_comprehensive_metrics


def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    embeddings = []
    pids = []
    iids = []
    paths = []
    with torch.no_grad():
        for imgs, product_ids, image_ids, rel_paths in loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            pids.extend([p.item() if hasattr(p, "item") else p for p in product_ids])
            iids.extend([i.item() if hasattr(i, "item") else i for i in image_ids])
            paths.extend(rel_paths)
    return np.vstack(embeddings).astype(np.float32), np.array(pids), np.array(iids), paths


def evaluate_split(
    query_embs: np.ndarray,
    query_pids: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_pids: np.ndarray,
    is_leave_one_out: bool = True,
    k_max: int = 10,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Queries gallery with query embeddings using exact inner-product search.
    If is_leave_one_out is True (query in gallery), excludes rank 0 (self).
    """
    dim = gallery_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(gallery_embs)

    k_search = k_max + 1 if is_leave_one_out else k_max
    distances, retrieved_indices = index.search(query_embs, k=k_search)

    if is_leave_one_out:
        # Exclude rank 0 (self)
        retrieved_indices_clean = retrieved_indices[:, 1:]
        distances_clean = distances[:, 1:]
    else:
        retrieved_indices_clean = retrieved_indices
        distances_clean = distances

    metrics = compute_comprehensive_metrics(query_pids, gallery_pids, retrieved_indices_clean)
    return metrics, retrieved_indices_clean, distances_clean


def generate_visual_retrieval_grid(
    query_info: List[dict],
    raw_dir: Path,
    output_path: Path,
):
    """
    Generates a visual comparison figure showing query images alongside their top-5 retrieved matches.
    """
    n_queries = len(query_info)
    fig, axes = plt.subplots(n_queries, 6, figsize=(18, 3.2 * n_queries), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.45)

    if n_queries == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, q in enumerate(query_info):
        # 1. Plot Query Image
        q_path = raw_dir / q["query_path"]
        q_img = Image.open(q_path).convert("RGB")
        axes[row_idx, 0].imshow(q_img)
        axes[row_idx, 0].set_title(
            f"QUERY\nPID: {q['query_pid']}\n({q['query_cat']})",
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
        )
        axes[row_idx, 0].axis("off")
        # Add thick dark border to query
        for spine in axes[row_idx, 0].spines.values():
            spine.set_visible(True)
            spine.set_color("#2563eb")
            spine.set_linewidth(3)

        # 2. Plot Top-5 Retrieved Matches
        for rank_idx, match in enumerate(q["matches"][:5]):
            col_idx = rank_idx + 1
            m_path = raw_dir / match["path"]
            m_img = Image.open(m_path).convert("RGB")
            ax = axes[row_idx, col_idx]
            ax.imshow(m_img)

            is_hit = match["is_hit"]
            border_color = "#16a34a" if is_hit else "#dc2626"
            status_text = "MATCH" if is_hit else "MISMATCH"
            score_pct = match["score"] * 100.0

            ax.set_title(
                f"#{rank_idx + 1} ({score_pct:.1f}%)\nPID: {match['pid']} | {match['cat']}\n[{status_text}]",
                fontsize=9,
                fontweight="bold",
                color=border_color,
            )
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(2.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Visual retrieval grid successfully saved to: {output_path}")


def main():
    cfg = load_config("configs/config.yaml")
    metadata_file = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    ckpt_path = PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / cfg["training"]["best_model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("COMPREHENSIVE METRIC-LEARNING EVALUATION PIPELINE")
    print(f"Device: {device} | Checkpoint: {ckpt_path.name}")
    print("=" * 75)

    # 1. Load Data
    test_dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split="test",
        is_triplet=False,
        image_size=cfg["preprocessing"]["image_size"],
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    full_dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split=None,
        is_triplet=False,
        image_size=cfg["preprocessing"]["image_size"],
    )
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    print(f"Test Split: {len(test_dataset)} images ({test_dataset.df['product_id'].nunique()} unseen classes)")
    print(f"Full Catalog Gallery: {len(full_dataset)} images ({full_dataset.df['product_id'].nunique()} total classes)")

    # 2. ImageNet Baseline Model
    print("\n[1/3] Extracting features with ImageNet-Pretrained Baseline...")
    baseline_model = ResNet50MetricNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=True,
        freeze_early_layers=False,
    ).to(device)
    base_test_embs, base_test_pids, _, _ = extract_features(baseline_model, test_loader, device)

    # 3. Fine-Tuned ResNet-50 MetricNet
    print("[2/3] Extracting features with Fine-Tuned ResNet-50 MetricNet...")
    trained_model = ResNet50MetricNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=False,
        freeze_early_layers=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    trained_model.to(device)
    trained_test_embs, trained_test_pids, trained_test_iids, trained_test_paths = extract_features(
        trained_model, test_loader, device
    )

    # Full catalog extraction for fine-tuned model
    trained_full_embs, trained_full_pids, trained_full_iids, trained_full_paths = extract_features(
        trained_model, full_loader, device
    )

    # 4. Compute Metrics (Unseen Test Set Leave-One-Out)
    print("\n[3/3] Evaluating Retrieval Metrics (Test Set Generalization)...")
    base_metrics, _, _ = evaluate_split(
        base_test_embs, base_test_pids, base_test_embs, base_test_pids, is_leave_one_out=True
    )
    trained_metrics, trained_ret_idx, trained_scores = evaluate_split(
        trained_test_embs, trained_test_pids, trained_test_embs, trained_test_pids, is_leave_one_out=True
    )

    # Also evaluate in Full Catalog scenario
    # In full catalog: each test query queries against all 200 catalog images. We map test query index to full catalog index to exclude self.
    full_index = faiss.IndexFlatIP(cfg["model"]["embedding_dim"])
    full_index.add(trained_full_embs)
    full_scores_all, full_ret_idx_all = full_index.search(trained_test_embs, k=11)
    # Exclude rank 0 (self-match)
    full_ret_idx_clean = full_ret_idx_all[:, 1:]
    full_catalog_metrics = compute_comprehensive_metrics(trained_test_pids, trained_full_pids, full_ret_idx_clean)

    # 5. Print Metric-Learning Evaluation Report
    print("\n" + "=" * 75)
    print("METRIC-LEARNING EVALUATION REPORT (UNSEEN IDENTITIES)")
    print("=" * 75)
    print(f"{'Metric':<18} | {'ImageNet Baseline':<20} | {'Fine-Tuned ResNet-50':<20} | {'Gain':<10}")
    print("-" * 75)
    metric_keys = ["HitRate@1", "HitRate@5", "Recall@5", "Recall@10", "Precision@5", "mAP@5"]
    for k in metric_keys:
        b_val = base_metrics[k]
        t_val = trained_metrics[k]
        gain = t_val - b_val
        unit = "%" if k != "mAP@5" else ""
        gain_str = f"+{gain:.2f}{unit}" if gain >= 0 else f"{gain:.2f}{unit}"
        print(f"{k:<18} | {b_val:>18}{unit} | {t_val:>18}{unit} | {gain_str:>10}")
    print("=" * 75)

    print("\n" + "=" * 75)
    print("FULL-CATALOG RETRIEVAL REPORT (40 Queries vs. 200 Catalog Images with 160 Distractors)")
    print("=" * 75)
    for k in metric_keys:
        val = full_catalog_metrics[k]
        unit = "%" if k != "mAP@5" else ""
        print(f"  {k:<18}: {val}{unit}")
    print("=" * 75)

    # 6. Generate Visual Retrieval Comparison
    # Pick 4 diverse query images from the test set (e.g. from different categories / products)
    df_test = test_dataset.df
    unique_products = df_test["product_id"].unique()
    sample_pids = unique_products[:4]

    query_info_list = []
    for pid in sample_pids:
        # Pick the first view of this product as query
        match_rows = df_test[df_test["product_id"] == pid]
        q_idx = match_rows.index[0] - df_test.index[0]  # Local index in test split
        q_row = match_rows.iloc[0]

        top_ret_indices = trained_ret_idx[q_idx, :5]
        top_scores = trained_scores[q_idx, :5]

        matches = []
        for r_idx, score in zip(top_ret_indices, top_scores):
            ret_pid = trained_test_pids[r_idx]
            ret_path = trained_test_paths[r_idx]
            ret_cat = df_test.iloc[r_idx]["category"]
            matches.append({
                "pid": ret_pid,
                "path": ret_path,
                "cat": ret_cat,
                "score": float(score),
                "is_hit": (ret_pid == pid),
            })

        query_info_list.append({
            "query_pid": pid,
            "query_path": q_row["relative_path"],
            "query_cat": q_row["category"],
            "matches": matches,
        })

    viz_output = PROJECT_ROOT / "visualizations" / "retrieval_visual_matches.png"
    generate_visual_retrieval_grid(query_info_list, raw_dir, viz_output)

    return {
        "baseline_test": base_metrics,
        "fine_tuned_test": trained_metrics,
        "full_catalog": full_catalog_metrics,
    }


if __name__ == "__main__":
    main()

