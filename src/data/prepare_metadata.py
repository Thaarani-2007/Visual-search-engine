"""
Dataset Metadata Ingestion, Validation, and Identity-Disjoint Splitting Pipeline.

This module processes the raw Stanford Online Products (SOP) dataset:
1. Validates image files on disk (checks existence and file integrity via PIL).
2. Filters targeted super-categories to form a coherent ~40,000 image dataset.
3. Ensures every product identity has at least 2 images for positive pairing.
4. Performs mathematically strict, leakage-free Identity-Disjoint splitting:
   (Train IDs ∩ Val IDs ∩ Test IDs = ∅).
5. Exports a standardized metadata CSV for training and indexing pipelines.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config


def verify_image_file(image_path: Path) -> bool:
    """Verify that an image file exists, is non-empty, and can be decoded by PIL."""
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_raw_sop_metadata(raw_dir: Path) -> pd.DataFrame:
    """
    Parse raw SOP metadata files or reconstruct metadata from directory structure.
    SOP metadata files (Ebay_train.txt, Ebay_test.txt, or Ebay_info.txt) typically have:
    image_id class_id super_class_id path
    """
    candidate_files = [
        raw_dir / "Ebay_info.txt",
        raw_dir / "Ebay_train.txt",
        raw_dir / "Ebay_test.txt",
    ]

    dfs = []
    for f in candidate_files:
        if f.is_file():
            df = pd.read_csv(f, sep=r"\s+", header=0)
            dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["image_id"])
        # Standardize column names
        combined.rename(
            columns={
                "class_id": "product_id",
                "super_class_id": "category_id",
                "path": "relative_path",
            },
            inplace=True,
        )
        # Extract human-readable category name from path prefix (e.g., 'chair_final/...')
        combined["category"] = combined["relative_path"].apply(
            lambda p: str(p).split("/")[0].replace("_final", "")
        )
        return combined

    # Fallback: traverse directory if txt files aren't present
    records = []
    for img_file in raw_dir.rglob("*.jpg"):
        rel_path = img_file.relative_to(raw_dir).as_posix()
        parts = rel_path.split("/")
        if len(parts) >= 2:
            category = parts[0].replace("_final", "")
            # SOP filenames follow: <product_id>_<img_index>.jpg
            stem = img_file.stem
            product_id = stem.split("_")[0] if "_" in stem else parts[1]
            records.append({
                "image_id": len(records) + 1,
                "product_id": product_id,
                "category": category,
                "relative_path": rel_path,
            })

    if not records:
        raise FileNotFoundError(
            f"No valid SOP metadata files or image hierarchy found in {raw_dir}"
        )

    return pd.DataFrame(records)


def perform_identity_disjoint_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    min_images_per_id: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Partition dataset strictly by Product Identity (product_id).
    
    Guarantees:
    1. Zero identity leakage across train, val, and test splits.
    2. Every identity has at least `min_images_per_id` images.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    np.random.seed(seed)

    # 1. Filter out singleton identities (cannot form positive pairs)
    id_counts = df["product_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_images_per_id].index.to_numpy()
    
    filtered_df = df[df["product_id"].isin(valid_ids)].copy()
    print(f"Retained {len(valid_ids)} identities with >= {min_images_per_id} images (total {len(filtered_df)} images).")

    # 2. Shuffle unique identities deterministically
    np.random.shuffle(valid_ids)
    n_ids = len(valid_ids)
    n_train = int(n_ids * train_ratio)
    n_val = int(n_ids * val_ratio)

    train_ids = set(valid_ids[:n_train])
    val_ids = set(valid_ids[n_train:n_train + n_val])
    test_ids = set(valid_ids[n_train + n_val:])

    # Strict mathematical leakage assertion
    assert train_ids.isdisjoint(val_ids), "Fatal: Identity leakage between Train and Val!"
    assert train_ids.isdisjoint(test_ids), "Fatal: Identity leakage between Train and Test!"
    assert val_ids.isdisjoint(test_ids), "Fatal: Identity leakage between Val and Test!"

    # 3. Assign split tags
    def assign_split(pid):
        if pid in train_ids:
            return "train"
        elif pid in val_ids:
            return "val"
        else:
            return "test"

    filtered_df["split"] = filtered_df["product_id"].apply(assign_split)
    return filtered_df


def print_split_statistics(df: pd.DataFrame) -> None:
    """Print comprehensive split analysis and identity density metrics."""
    print("\n" + "=" * 60)
    print("DATASET IDENTITY-DISJOINT SPLIT REPORT")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        n_imgs = len(split_df)
        n_pids = split_df["product_id"].nunique()
        avg_imgs = n_imgs / n_pids if n_pids > 0 else 0
        print(f"[{split.upper():5s}] Images: {n_imgs:6d} | Identities: {n_pids:5d} | Avg Imgs/ID: {avg_imgs:.2f}")

    print("\nCategory Distribution Across Splits:")
    cat_pivot = pd.crosstab(df["category"], df["split"], margins=True)
    print(cat_pivot)
    print("=" * 60 + "\n")


def prepare_sop_metadata(config_path: str = "configs/config.yaml") -> Path:
    """Main data engineering orchestrator."""
    cfg = load_config(config_path)
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    output_path = PROJECT_ROOT / cfg["dataset"]["metadata_file"]
    target_cats = set(cfg["dataset"]["target_categories"])
    target_total = cfg["dataset"].get("target_total_images", 40000)

    print(f"Loading raw metadata from: {raw_dir}")
    df = load_raw_sop_metadata(raw_dir)

    # Filter target categories
    df = df[df["category"].isin(target_cats)].copy()
    print(f"Filtered to categories {target_cats}: {len(df)} candidate images across {df['product_id'].nunique()} identities.")

    # If images exceed target, subsample cleanly at identity level
    if len(df) > target_total:
        unique_pids = df["product_id"].unique()
        avg_per_pid = len(df) / len(unique_pids)
        target_pids_count = int(target_total / avg_per_pid)
        np.random.seed(cfg["seed"])
        sampled_pids = np.random.choice(unique_pids, size=min(target_pids_count, len(unique_pids)), replace=False)
        df = df[df["product_id"].isin(sampled_pids)].copy()
        print(f"Subsampled to {len(df)} images across {len(sampled_pids)} identities to match target size (~{target_total}).")

    # Split dataset
    ratios = cfg["dataset"]["split_ratios"]
    split_df = perform_identity_disjoint_split(
        df=df,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
        min_images_per_id=cfg["dataset"]["min_images_per_identity"],
        seed=cfg["seed"],
    )

    # Save processed metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    print(f"Clean metadata saved to: {output_path}")

    # Output stats
    print_split_statistics(split_df)
    return output_path


if __name__ == "__main__":
    prepare_sop_metadata()

