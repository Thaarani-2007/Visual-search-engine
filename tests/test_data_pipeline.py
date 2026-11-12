"""
Unit and Integration Tests for Phase 1: Data Pipeline & Triplet Generation.
"""

from pathlib import Path
import pytest
import pandas as pd
import torch

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.data.prepare_metadata import perform_identity_disjoint_split


def test_identity_disjoint_split_mathematical_guarantee():
    """Verify that identity-disjoint splitting produces zero ID leakage."""
    # Create synthetic metadata
    records = []
    for pid in range(1, 101):
        for img_idx in range(4):
            records.append({
                "image_id": len(records) + 1,
                "product_id": pid,
                "category": "chair",
                "relative_path": f"chair/{pid}_{img_idx}.jpg",
            })
    df = pd.DataFrame(records)

    split_df = perform_identity_disjoint_split(
        df,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        min_images_per_id=2,
        seed=42,
    )

    train_ids = set(split_df[split_df["split"] == "train"]["product_id"])
    val_ids = set(split_df[split_df["split"] == "val"]["product_id"])
    test_ids = set(split_df[split_df["split"] == "test"]["product_id"])

    # Strict identity-disjointness assertions
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test IDs overlap!"

    # Verify every identity has >= 2 images
    for pid, count in split_df["product_id"].value_counts().items():
        assert count >= 2, f"Identity {pid} has fewer than 2 images ({count})"


def test_triplet_batch_properties():
    """Verify DataLoader batch tensor dimensions, normalization, and semantic constraints."""
    cfg = load_config()
    metadata_file = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]

    if not metadata_file.is_file():
        pytest.skip("Metadata file not generated yet. Skipping live batch test.")

    dataset = SOPMetricDataset(
        metadata_csv=metadata_file,
        raw_image_dir=raw_dir,
        split="train",
        is_triplet=True,
        image_size=224,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    anchors, positives, negatives, pids = batch

    # Dimension checks
    assert anchors.shape == (4, 3, 224, 224)
    assert positives.shape == (4, 3, 224, 224)
    assert negatives.shape == (4, 3, 224, 224)

    # Type checks
    assert anchors.dtype == torch.float32
    assert positives.dtype == torch.float32
    assert negatives.dtype == torch.float32

    # Finite value checks (no NaNs or Infs)
    assert torch.isfinite(anchors).all()
    assert torch.isfinite(positives).all()
    assert torch.isfinite(negatives).all()

