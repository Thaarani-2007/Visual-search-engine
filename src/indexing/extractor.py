"""
High-Throughput Offline Embedding Extraction Pipeline.

Extracts 512-D L2-normalized embeddings for the full image gallery using the trained
ResNet-50 metric learning checkpoint.

Outputs:
1. data/index/embeddings.npy: Float32 NumPy array of shape [N, 512].
2. data/index/image_ids.npy: Int/Str NumPy array of shape [N].
3. data/index/metadata_lookup.json: Inverted mapping from FAISS integer index [0..N-1]
   to complete image metadata (image_id, product_id, relative_path, category).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.models.siamese_net import ResNet50MetricNet


class OfflineEmbeddingExtractor:
    """Extracts and persists dense vector embeddings and metadata for FAISS indexing."""

    def __init__(self, config_path: str = "configs/config.yaml", checkpoint_path: str = None, device: str = None):
        self.cfg = load_config(config_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Extractor initialized on compute device: {self.device}")

        # Resolve checkpoint path
        if checkpoint_path is None:
            ckpt_path = PROJECT_ROOT / self.cfg["training"]["checkpoint_dir"] / self.cfg["training"]["best_model_name"]
        else:
            ckpt_path = Path(checkpoint_path)

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}. Run training first.")

        # Load model
        self.model = ResNet50MetricNet(
            embedding_dim=self.cfg["model"]["embedding_dim"],
            pretrained=False,
            freeze_early_layers=False,
        )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded trained metric model from: {ckpt_path}")

        # Index output directory
        self.index_dir = PROJECT_ROOT / self.cfg["indexing"]["index_dir"]
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def extract_dataset_embeddings(
        self, split: str = None, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, dict]]:
        """
        Extracts embeddings for all images in the specified split (or full dataset if split is None).
        Returns:
            embeddings: [N, 512] float32 array
            image_ids: [N] array of image IDs
            metadata_lookup: dict mapping faiss_index -> metadata
        """
        metadata_file = PROJECT_ROOT / self.cfg["dataset"]["metadata_file"]
        raw_dir = PROJECT_ROOT / self.cfg["dataset"]["raw_dir"]

        df = pd.read_csv(metadata_file)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        print(f"\nExtracting embeddings for {len(df)} images (split='{split}')...")

        # Use non-triplet extraction mode (single images with deterministic evaluation transforms)
        dataset = SOPMetricDataset(
            metadata_csv=metadata_file,
            raw_image_dir=raw_dir,
            split=split if split else "test",
            is_triplet=False,
            image_size=self.cfg["preprocessing"]["image_size"],
        )
        dataset.df = df  # ensure exact filtered frame

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        embeddings_list = []
        metadata_lookup = {}
        index_counter = 0

        with torch.no_grad():
            for images, pids, img_ids, rel_paths in tqdm(loader, desc="Extracting"):
                images = images.to(self.device, non_blocking=True)
                # Forward pass through model produces L2-normalized embeddings
                batch_embeddings = self.model(images).cpu().numpy()

                embeddings_list.append(batch_embeddings)

                for i in range(len(batch_embeddings)):
                    pid_val = pids[i].item() if hasattr(pids[i], "item") else pids[i]
                    img_id_val = img_ids[i].item() if hasattr(img_ids[i], "item") else img_ids[i]

                    metadata_lookup[index_counter] = {
                        "faiss_id": index_counter,
                        "image_id": int(img_id_val) if str(img_id_val).isdigit() else str(img_id_val),
                        "product_id": int(pid_val) if str(pid_val).isdigit() else str(pid_val),
                        "relative_path": rel_paths[i],
                    }
                    index_counter += 1

        all_embeddings = np.vstack(embeddings_list).astype(np.float32)
        all_image_ids = np.array([m["image_id"] for m in metadata_lookup.values()])

        print(f"Extraction complete. Embeddings matrix shape: {all_embeddings.shape}, dtype: {all_embeddings.dtype}")
        return all_embeddings, all_image_ids, metadata_lookup

    def save_extracted_artifacts(
        self, embeddings: np.ndarray, image_ids: np.ndarray, metadata_lookup: Dict[int, dict]
    ) -> Tuple[Path, Path, Path]:
        """Save extracted numpy matrices and JSON metadata lookup file."""
        emb_path = self.index_dir / "embeddings.npy"
        ids_path = self.index_dir / "image_ids.npy"
        meta_path = self.index_dir / "metadata_lookup.json"

        np.save(emb_path, embeddings)
        np.save(ids_path, image_ids)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_lookup, f, indent=2)

        print(f"Saved embeddings to:      {emb_path} ({emb_path.stat().st_size / (1024*1024):.2f} MB)")
        print(f"Saved image IDs to:       {ids_path}")
        print(f"Saved metadata lookup to: {meta_path}")

        return emb_path, ids_path, meta_path


def run_extraction():
    extractor = OfflineEmbeddingExtractor()
    # Extract all indexed images
    embeddings, image_ids, metadata_lookup = extractor.extract_dataset_embeddings(split=None)
    extractor.save_extracted_artifacts(embeddings, image_ids, metadata_lookup)


if __name__ == "__main__":
    run_extraction()

