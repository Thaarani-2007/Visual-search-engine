"""
ResNet-50 Deep Metric Learning Architecture.

Features:
- ImageNet-pretrained ResNet-50 backbone.
- Configurable layer freezing (preserves generic low-level vision primitives).
- Metric projection head (Linear 2048 -> 512 + BatchNorm1d).
- L2 Hypersphere Normalization layer.
- Dual forward modes:
    - forward_triplet(anchor, positive, negative): for training with Triplet Loss.
    - forward(x): for single-query inference and FAISS gallery embedding extraction.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class MetricProjectionHead(nn.Module):
    """
    Projects 2048-D backbone feature representations into a 512-D metric space.
    BatchNorm1d centers and normalizes the variance before unit-sphere projection,
    drastically reducing the risk of dimensional collapse.
    """

    def __init__(self, in_features: int = 2048, embedding_dim: int = 512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ResNet50MetricNet(nn.Module):
    """
    Production-grade Metric Learning Network based on ResNet-50.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        freeze_until: str = "layer3",  # Freezes conv1, bn1, layer1, layer2; leaves layer3, layer4 trainable
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_early_layers = freeze_early_layers
        self.freeze_until = freeze_until

        # 1. Load ResNet-50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # 2. Extract feature extractor components (remove original 1000-class fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Metric Projection Head
        self.head = nn.Sequential(
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # 4. Apply layer freezing policy
        if self.freeze_early_layers:
            self._apply_layer_freezing()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Flexible state_dict loader that handles head.projection vs head key differences."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("head.projection."):
                # Map head.projection.0.weight -> head.0.weight
                new_key = k.replace("head.projection.", "head.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict=strict)

    def _apply_layer_freezing(self) -> None:
        """
        Freezes early layers (conv1, bn1, layer1, layer2) to preserve generic ImageNet features.
        Leaves later layers (layer3, layer4) trainable for task-specific fine-tuning.
        """
        layers_to_freeze = [self.conv1, self.bn1, self.layer1, self.layer2]
        for module in layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return counts of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": round((trainable / total) * 100, 2),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input image through ResNet-50 backbone up to adaptive pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        return x

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single image forward pass:
        Input [B, 3, 224, 224] -> Backbone [B, 2048] -> Projection [B, 512] -> L2 Normalized [B, 512]
        """
        features = self.extract_features(x)
        projected = self.head(features)
        # Unit Hypersphere Projection (L2 Normalization with epsilon to avoid division by zero)
        normalized = F.normalize(projected, p=2, dim=1, eps=1e-12)
        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for single images (inference and offline index extraction)."""
        return self.forward_one(x)

    def forward_triplet(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Triplet forward pass for Siamese metric learning training.
        Returns:
            a_emb: [B, 512]
            p_emb: [B, 512]
            n_emb: [B, 512]
        """
        a_emb = self.forward_one(anchor)
        p_emb = self.forward_one(positive)
        n_emb = self.forward_one(negative)
        return a_emb, p_emb, n_emb

