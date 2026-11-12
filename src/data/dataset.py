"""
PyTorch Dataset Implementation for Deep Metric Learning (Siamese / Triplet Architecture).

Features:
- Inverted index for O(1) Anchor-Positive-Negative retrieval.
- Configurable negative sampling (Global Random vs Within-Category Semi-Hard).
- Disjoint train/val/test augmentation pipelines.
- Supports both Triplet mode (for training) and Single-image mode (for embedding extraction/indexing).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """
    Produce appropriate torchvision transformation pipeline based on split.
    
    Training: Stochastic augmentations (Crop, Flip, Jitter) to build visual invariance.
    Validation / Test / Indexing: Deterministic Resize + CenterCrop for reproducible embeddings.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Deterministic evaluation pipeline
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # e.g., 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class SOPMetricDataset(Dataset):
    """
    Production-grade Dataset for Stanford Online Products (SOP).
    
    Operates in two modes:
    1. is_triplet=True (Default): Returns (anchor, positive, negative, label) for metric learning.
    2. is_triplet=False: Returns (image, product_id, image_id, relative_path) for embedding extraction & FAISS indexing.
    """

    def __init__(
        self,
        metadata_csv: Union[str, Path],
        raw_image_dir: Union[str, Path],
        split: str = "train",
        is_triplet: bool = True,
        negative_sampling_strategy: str = "semi_hard",  # "random" or "semi_hard" (within-category)
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.metadata_csv = Path(metadata_csv)
        self.raw_image_dir = Path(raw_image_dir)
        self.split = split
        self.is_triplet = is_triplet
        self.strategy = negative_sampling_strategy

        # Load metadata and filter by split
        df = pd.read_csv(self.metadata_csv)
        if split is not None and split != "all" and "split" in df.columns:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in {metadata_csv}")

        split_for_trans = split if split in ["train", "val", "test"] else "test"
        self.transform = transform if transform else get_transforms(split=split_for_trans, image_size=image_size)

        # Precompute inverted indices for O(1) sampling
        self.id_to_indices: Dict[Union[int, str], List[int]] = {}
        self.cat_to_ids: Dict[str, List[Union[int, str]]] = {}
        self.unique_ids: List[Union[int, str]] = list(self.df["product_id"].unique())

        for idx, row in self.df.iterrows():
            pid = row["product_id"]
            cat = row["category"]
            self.id_to_indices.setdefault(pid, []).append(idx)
            if cat not in self.cat_to_ids:
                self.cat_to_ids[cat] = []
            if pid not in self.cat_to_ids[cat]:
                self.cat_to_ids[cat].append(pid)

        # Convert to numpy arrays for high-performance random indexing
        self.id_to_indices_np = {k: np.array(v) for k, v in self.id_to_indices.items()}
        self.unique_ids_np = np.array(self.unique_ids)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, relative_path: str) -> Image.Image:
        """Safely load and convert image to RGB."""
        full_path = self.raw_image_dir / relative_path
        with Image.open(full_path) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int):
        if not self.is_triplet:
            # Single image extraction mode (used for FAISS indexing & evaluation)
            row = self.df.iloc[index]
            img = self._load_image(row["relative_path"])
            if self.transform:
                img = self.transform(img)
            return img, row["product_id"], row["image_id"], row["relative_path"]

        # --- Triplet Generation Mode ---
        anchor_row = self.df.iloc[index]
        anchor_pid = anchor_row["product_id"]
        anchor_cat = anchor_row["category"]

        # 1. Sample Positive: Pick a different image sharing the exact same product identity
        positive_candidate_indices = self.id_to_indices_np[anchor_pid]
        if len(positive_candidate_indices) > 1:
            # Exclude self
            valid_positives = positive_candidate_indices[positive_candidate_indices != index]
            pos_idx = np.random.choice(valid_positives)
        else:
            # Fallback if only 1 image exists: use self (augmentation creates variance)
            pos_idx = index
        pos_row = self.df.iloc[pos_idx]

        # 2. Sample Negative
        if self.strategy == "semi_hard" and anchor_cat in self.cat_to_ids and len(self.cat_to_ids[anchor_cat]) > 1:
            # Within-category negative: same category (e.g. chair), but DIFFERENT product identity
            cat_pids = self.cat_to_ids[anchor_cat]
            neg_pid = np.random.choice(cat_pids)
            while neg_pid == anchor_pid:
                neg_pid = np.random.choice(cat_pids)
        else:
            # Global random negative: any different identity in dataset
            neg_pid = np.random.choice(self.unique_ids_np)
            while neg_pid == anchor_pid:
                neg_pid = np.random.choice(self.unique_ids_np)

        neg_candidate_indices = self.id_to_indices_np[neg_pid]
        neg_idx = np.random.choice(neg_candidate_indices)
        neg_row = self.df.iloc[neg_idx]

        # Load RGB images
        anchor_img = self._load_image(anchor_row["relative_path"])
        pos_img = self._load_image(pos_row["relative_path"])
        neg_img = self._load_image(neg_row["relative_path"])

        # Apply independent transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return (
            anchor_img,
            pos_img,
            neg_img,
            anchor_pid,
        )

