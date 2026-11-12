"""
Embedding Space 2D t-SNE Visualization Script.

Visualizes the 512-dimensional hypersphere representations before and after
metric learning using t-SNE dimensionality reduction.

Outputs:
- visualizations/embedding_tsne.png: Side-by-side 2D cluster comparison.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.models.siamese_net import ResNet50MetricNet
from scripts.evaluate_retrieval import extract_features_for_eval


def visualize_embedding_clusters(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    metadata_file = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    ckpt_path = PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / cfg["training"]["best_model_name"]
    vis_dir = PROJECT_ROOT / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_file = vis_dir / "embedding_tsne.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating t-SNE Embedding Space Visualization...")

    # Load test split
    test_dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split="test",
        is_triplet=False,
        image_size=cfg["preprocessing"]["image_size"],
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 1. Baseline embeddings
    baseline_model = ResNet50MetricNet(embedding_dim=512, pretrained=True, freeze_early_layers=False).to(device)
    base_embs, labels = extract_features_for_eval(baseline_model, test_loader, device)

    # 2. Trained embeddings
    trained_model = ResNet50MetricNet(embedding_dim=512, pretrained=False, freeze_early_layers=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    trained_model.to(device)
    trained_embs, _ = extract_features_for_eval(trained_model, test_loader, device)

    # Filter top 8 identities for clean, interpretable visual separation
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_labels = unique_labels[np.argsort(-counts)[:8]]
    mask = np.isin(labels, top_labels)

    filtered_base = base_embs[mask]
    filtered_trained = trained_embs[mask]
    filtered_labels = labels[mask]

    # Map labels to 0..K for discrete color palettes
    label_to_idx = {l: i for i, l in enumerate(top_labels)}
    color_indices = np.array([label_to_idx[l] for l in filtered_labels])

    # Run t-SNE
    perplexity = min(15, len(filtered_base) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=cfg["seed"], init="pca", learning_rate="auto")
    base_2d = tsne.fit_transform(filtered_base)
    trained_2d = tsne.fit_transform(filtered_trained)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap("tab10", len(top_labels))

    # Baseline Plot
    scatter1 = axes[0].scatter(
        base_2d[:, 0], base_2d[:, 1], c=color_indices, cmap=cmap, s=60, alpha=0.85, edgecolors="none"
    )
    axes[0].set_title("ImageNet Baseline (Untrained Metric Head)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, alpha=0.2)

    # Trained Metric Learning Plot
    scatter2 = axes[1].scatter(
        trained_2d[:, 0], trained_2d[:, 1], c=color_indices, cmap=cmap, s=60, alpha=0.85, edgecolors="none"
    )
    axes[1].set_title("Fine-Tuned ResNet-50 (Triplet Loss + Hard Negatives)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"t-SNE visualization saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    visualize_embedding_clusters()
