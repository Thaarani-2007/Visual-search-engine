"""
Metric Learning Training and Validation Pipeline.

Features:
- Adam optimizer with weight decay and learning rate scheduling.
- Online Semi-Hard Negative Mining.
- Gradient norm clipping to prevent gradient explosion.
- Validation loop with embedding collapse detection (variance monitoring).
- Best checkpoint saving and training history visualization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import SOPMetricDataset
from src.models.siamese_net import ResNet50MetricNet
from src.training.losses import OnlineTripletLoss


class MetricTrainer:
    """Production Trainer for ResNet-50 Siamese / Triplet Deep Metric Learning."""

    def __init__(self, config_path: str = "configs/config.yaml", device: str = None):
        self.cfg = load_config(config_path)

        # Device selection: CUDA GPU if available (Kaggle/Local), else CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Trainer initialized on compute device: {self.device}")

        # Model
        self.model = ResNet50MetricNet(
            embedding_dim=self.cfg["model"]["embedding_dim"],
            pretrained=self.cfg["model"]["pretrained"],
            freeze_early_layers=self.cfg["model"]["freeze_early_layers"],
        ).to(self.device)

        # Loss function
        self.criterion = OnlineTripletLoss(
            margin=self.cfg["training"]["margin"],
            semi_hard=self.cfg["training"]["hard_negative_mining"],
        )

        # Optimizer: only optimize parameters with requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(
            trainable_params,
            lr=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )

        # Learning rate scheduler
        self.epochs = self.cfg["training"]["epochs"]
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        # Checkpoints & Artifacts
        self.checkpoint_dir = PROJECT_ROOT / self.cfg["training"]["checkpoint_dir"]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / self.cfg["training"]["best_model_name"]

        # History tracker
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_separation_gap": [],
            "val_separation_gap": [],
            "val_embedding_variance": [],
            "lr": [],
        }

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Construct Train and Validation DataLoaders."""
        metadata_file = PROJECT_ROOT / self.cfg["dataset"]["metadata_file"]
        raw_dir = PROJECT_ROOT / self.cfg["dataset"]["raw_dir"]
        batch_size = self.cfg["preprocessing"]["batch_size"]
        num_workers = self.cfg["preprocessing"].get("num_workers", 0)

        train_dataset = SOPMetricDataset(
            metadata_csv=metadata_file,
            raw_image_dir=raw_dir,
            split="train",
            is_triplet=True,
            image_size=self.cfg["preprocessing"]["image_size"],
        )

        val_dataset = SOPMetricDataset(
            metadata_csv=metadata_file,
            raw_image_dir=raw_dir,
            split="val",
            is_triplet=True,
            image_size=self.cfg["preprocessing"]["image_size"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch with online hard-negative mining."""
        self.model.train()
        total_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        total_active_ratio = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for anchors, positives, negatives, pids in pbar:
            anchors = anchors.to(self.device, non_blocking=True)
            positives = positives.to(self.device, non_blocking=True)
            negatives = negatives.to(self.device, non_blocking=True)

            # Convert string or int pids to long tensor
            if isinstance(pids[0], str):
                unique_pids = list(set(pids))
                pid_map = {pid: i for i, pid in enumerate(unique_pids)}
                labels = torch.tensor([pid_map[p] for p in pids], device=self.device)
            else:
                labels = torch.as_tensor(pids, device=self.device)

            self.optimizer.zero_grad()

            # Forward pass through Siamese network
            a_emb, p_emb, n_emb = self.model.forward_triplet(anchors, positives, negatives)

            # Compute Online Triplet Loss
            loss, metrics = self.criterion(a_emb, p_emb, labels, negative_embeddings=n_emb)

            # Backpropagation with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += metrics["loss"]
            total_pos_dist += metrics["mean_pos_dist"]
            total_neg_dist += metrics["mean_neg_dist"]
            total_active_ratio += metrics["active_triplets_ratio"]

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "gap": f"{metrics['separation_gap']:.3f}",
                "active%": f"{metrics['active_triplets_ratio'] * 100:.1f}%",
            })

        n_batches = len(train_loader)
        return {
            "loss": total_loss / n_batches,
            "mean_pos_dist": total_pos_dist / n_batches,
            "mean_neg_dist": total_neg_dist / n_batches,
            "separation_gap": (total_neg_dist - total_pos_dist) / n_batches,
            "active_ratio": total_active_ratio / n_batches,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation and detect potential embedding collapse."""
        self.model.eval()
        total_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        all_embeddings = []

        with torch.no_grad():
            for anchors, positives, negatives, pids in val_loader:
                anchors = anchors.to(self.device, non_blocking=True)
                positives = positives.to(self.device, non_blocking=True)
                negatives = negatives.to(self.device, non_blocking=True)

                if isinstance(pids[0], str):
                    unique_pids = list(set(pids))
                    pid_map = {pid: i for i, pid in enumerate(unique_pids)}
                    labels = torch.tensor([pid_map[p] for p in pids], device=self.device)
                else:
                    labels = torch.as_tensor(pids, device=self.device)

                a_emb, p_emb, n_emb = self.model.forward_triplet(anchors, positives, negatives)
                loss, metrics = self.criterion(a_emb, p_emb, labels, negative_embeddings=n_emb)

                total_loss += metrics["loss"]
                total_pos_dist += metrics["mean_pos_dist"]
                total_neg_dist += metrics["mean_neg_dist"]
                all_embeddings.append(a_emb.cpu())

        n_batches = max(len(val_loader), 1)
        mean_pos = total_pos_dist / n_batches
        mean_neg = total_neg_dist / n_batches
        gap = mean_neg - mean_pos

        # Embedding Collapse Detection: compute mean per-dimension std across all validation vectors
        all_embs_tensor = torch.cat(all_embeddings, dim=0)  # [N_val, 512]
        dim_variance = torch.std(all_embs_tensor, dim=0).mean().item()

        if dim_variance < 0.01:
            print("\n[WARNING] Potential Embedding Collapse detected! Dim variance:", dim_variance)

        return {
            "val_loss": total_loss / n_batches,
            "val_pos_dist": mean_pos,
            "val_neg_dist": mean_neg,
            "val_separation_gap": gap,
            "val_embedding_variance": dim_variance,
        }

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = True) -> None:
        """Export clean PyTorch model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.cfg,
        }
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"  ★ New best model checkpoint saved to: {self.best_model_path} (Val Loss: {val_loss:.4f})")

    def plot_history(self, output_path: Path = None) -> None:
        """Render and save loss and separation curves."""
        if output_path is None:
            output_path = PROJECT_ROOT / "visualizations" / "training_curves.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        # Subplot 1: Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history["train_loss"], label="Train Triplet Loss", color="royalblue", lw=2)
        plt.plot(epochs_range, self.history["val_loss"], label="Val Triplet Loss", color="darkorange", lw=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Triplet Margin Loss (Margin = 0.2)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Separation gap [d(a, n) - d(a, p)]
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.history["train_separation_gap"], label="Train Gap", color="green", lw=2)
        plt.plot(epochs_range, self.history["val_separation_gap"], label="Val Gap", color="purple", lw=2)
        plt.axhline(y=0.2, color="gray", linestyle="--", label="Target Margin (0.2)")
        plt.xlabel("Epoch")
        plt.ylabel("Separation Gap: d(a,n) - d(a,p)")
        plt.title("Metric Separation Dynamics")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Training visualization saved to: {output_path}")

    def train(self, num_epochs: int = None) -> Dict[str, List[float]]:
        """Full training loop execution."""
        if num_epochs is None:
            num_epochs = self.epochs

        train_loader, val_loader = self.get_dataloaders()
        best_val_loss = float("inf")

        print(f"\nBeginning Metric Learning Training for {num_epochs} epochs...")
        print("=" * 70)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_separation_gap"].append(train_metrics["separation_gap"])
            self.history["val_separation_gap"].append(val_metrics["val_separation_gap"])
            self.history["val_embedding_variance"].append(val_metrics["val_embedding_variance"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch:2d}/{num_epochs:2d} | "
                f"Train Loss: {train_metrics['loss']:.4f} (Gap: {train_metrics['separation_gap']:.3f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (Gap: {val_metrics['val_separation_gap']:.3f}) | "
                f"Active Triplets: {train_metrics['active_ratio'] * 100:.1f}%"
            )

            # Checkpoint saving
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch=epoch, val_loss=best_val_loss, is_best=True)

        self.plot_history()
        return self.history


if __name__ == "__main__":
    trainer = MetricTrainer()
    trainer.train(num_epochs=1)
