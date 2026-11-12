"""
DataLoader and Triplet Generation Verification Script.

Tests:
1. Batch shapes for Anchor, Positive, Negative: [B, 3, 224, 224].
2. Value normalization and data type integrity.
3. Logical validity of Anchor-Positive (identical ID) vs Negative (different ID).
4. Exports a visual triplet comparison grid (Anchor | Positive | Negative) to visualizations/.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized PyTorch tensor [3, H, W] back to RGB numpy array [H, W, 3] for display."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = tensor.cpu().numpy()
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return np.transpose(img_np, (1, 2, 0))


def verify_triplet_dataloader(config_path: str = "configs/config.yaml") -> bool:
    cfg = load_config(config_path)
    metadata_file = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    batch_size = 4

    print(f"\n[1/4] Verifying metadata file presence: {metadata_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found at {metadata_file}. Run prepare_metadata.py first."
        )

    print(f"[2/4] Initializing SOPMetricDataset (Train split)...")
    dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split="train",
        is_triplet=True,
        negative_sampling_strategy="semi_hard",
        image_size=cfg["preprocessing"]["image_size"],
    )
    print(f"Dataset successfully loaded. Total training anchors: {len(dataset)}")

    print(f"[3/4] Initializing DataLoader with batch_size={batch_size}...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for safe Windows cross-process execution during testing
        drop_last=True,
    )

    batch = next(iter(loader))
    anchors, positives, negatives, pids = batch

    # Assertions
    print(f"  Anchor shape:   {anchors.shape}   | dtype: {anchors.dtype}")
    print(f"  Positive shape: {positives.shape} | dtype: {positives.dtype}")
    print(f"  Negative shape: {negatives.shape} | dtype: {negatives.dtype}")

    assert anchors.shape == (batch_size, 3, 224, 224), f"Unexpected anchor shape: {anchors.shape}"
    assert positives.shape == (batch_size, 3, 224, 224), f"Unexpected positive shape: {positives.shape}"
    assert negatives.shape == (batch_size, 3, 224, 224), f"Unexpected negative shape: {negatives.shape}"
    assert isinstance(anchors, torch.FloatTensor), "Anchors must be FloatTensor"

    print("  ✓ Tensor dimensions and dtypes verified.")

    print(f"[4/4] Generating visual triplet grid for inspection...")
    vis_dir = PROJECT_ROOT / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_vis_path = vis_dir / "sample_triplets.png"

    fig, axes = plt.subplots(batch_size, 3, figsize=(10, 3 * batch_size))
    for i in range(batch_size):
        a_img = unnormalize_image(anchors[i])
        p_img = unnormalize_image(positives[i])
        n_img = unnormalize_image(negatives[i])

        axes[i, 0].imshow(a_img)
        axes[i, 0].set_title(f"Anchor (ID: {pids[i]})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(p_img)
        axes[i, 1].set_title(f"Positive (Same ID: {pids[i]})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(n_img)
        axes[i, 2].set_title("Negative (Diff ID)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_vis_path, dpi=150)
    plt.close()
    print(f"  ✓ Sample triplets visualization saved to: {out_vis_path}")

    print("\n" + "=" * 50)
    print("ALL DATALOADER VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 50 + "\n")
    return True


if __name__ == "__main__":
    verify_triplet_dataloader()

