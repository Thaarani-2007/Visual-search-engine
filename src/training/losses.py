"""
Loss Functions and Online Negative Mining for Deep Metric Learning.

Implements:
1. OnlineSemiHardNegativeTripletLoss: Vectorized in-batch semi-hard / hard negative mining.
2. Diagnostic metric computations: positive distance, negative distance, active triplet ratio.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(embeddings: torch.Tensor, squared: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """
    Vectorized computation of all-pairs Euclidean distance matrix D in R^{B x B}.
    For L2-normalized embeddings: ||u - v||_2^2 = 2 - 2 * <u, v>.
    """
    # Dot product matrix: [B, B]
    dot_product = torch.matmul(embeddings, embeddings.t())
    # Numerical clamp between -1.0 and 1.0
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u, v> = 2 - 2<u, v> for unit vectors
    dist_sq = 2.0 - 2.0 * dot_product
    # Clamp below to avoid negative values from floating point error
    dist_sq = torch.clamp(dist_sq, min=0.0)

    if squared:
        return dist_sq

    # Safe square root with epsilon to prevent NaN gradients at zero distance
    return torch.sqrt(dist_sq + eps)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Margin Loss with in-batch semi-hard negative mining.

    For each anchor-positive pair (a_i, p_i) in the batch, it searches the batch
    for the most challenging negative n_j (where label_j != label_i) that satisfies
    either:
        Semi-Hard: d(a, p) < d(a, n) < d(a, p) + margin
        Hard Fallback: argmin_{n} d(a, n)
    """

    def __init__(self, margin: float = 0.2, semi_hard: bool = True):
        super().__init__()
        self.margin = margin
        self.semi_hard = semi_hard

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_embeddings: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            anchor_embeddings: [B, 512]
            positive_embeddings: [B, 512]
            labels: [B] product identity labels
            negative_embeddings: Optional [B, 512] pre-sampled negative embeddings.
        Returns:
            loss: scalar loss tensor
            metrics: dict with diagnostic metrics
        """
        batch_size = anchor_embeddings.size(0)

        # 1. Compute anchor-positive pairwise distance for each sample: [B]
        # d(a_i, p_i) = sqrt(2 - 2 * <a_i, p_i>)
        ap_dot = (anchor_embeddings * positive_embeddings).sum(dim=1)
        ap_dot = torch.clamp(ap_dot, -1.0, 1.0)
        d_ap = torch.sqrt(torch.clamp(2.0 - 2.0 * ap_dot, min=1e-12))  # [B]

        if negative_embeddings is not None and not self.semi_hard:
            # Simple offline/pre-sampled negative computation
            an_dot = (anchor_embeddings * negative_embeddings).sum(dim=1)
            an_dot = torch.clamp(an_dot, -1.0, 1.0)
            d_an = torch.sqrt(torch.clamp(2.0 - 2.0 * an_dot, min=1e-12))
        else:
            # 2. In-Batch Online Negative Mining
            # Stack all available embeddings in batch to form negative search pool: [2B, 512]
            pool_embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
            pool_labels = torch.cat([labels, labels], dim=0)  # [2B]

            # Pairwise distance matrix between Anchors [B] and Pool [2B]: Shape [B, 2B]
            dist_matrix = pairwise_distance(
                torch.cat([anchor_embeddings, pool_embeddings], dim=0)
            )[:batch_size, batch_size:]  # [B, 2B]

            # Mask of valid negatives: pool_label != anchor_label
            anchor_labels_expanded = labels.unsqueeze(1)  # [B, 1]
            pool_labels_expanded = pool_labels.unsqueeze(0)  # [1, 2B]
            is_negative = anchor_labels_expanded != pool_labels_expanded  # [B, 2B]

            # Selected negative distance per anchor: [B]
            selected_d_an = []

            for i in range(batch_size):
                neg_mask_i = is_negative[i]
                if not neg_mask_i.any():
                    # No negative in batch with different label (rare edge case: batch has 1 class)
                    selected_d_an.append(d_ap[i] + self.margin)
                    continue

                neg_dists_i = dist_matrix[i, neg_mask_i]
                d_ap_i = d_ap[i]

                # Semi-Hard criterion: d(a, p) < d(a, n) < d(a, p) + margin
                semi_hard_mask = (neg_dists_i > d_ap_i) & (neg_dists_i < d_ap_i + self.margin)

                if semi_hard_mask.any():
                    # Pick the hardest among semi-hard negatives (minimum distance among semi-hard)
                    chosen_neg = torch.min(neg_dists_i[semi_hard_mask])
                else:
                    # Fallback to the hardest negative overall (closest negative)
                    chosen_neg = torch.min(neg_dists_i)

                selected_d_an.append(chosen_neg)

            d_an = torch.stack(selected_d_an)  # [B]

        # 3. Compute Triplet Margin Loss: max(d(a, p) - d(a, n) + margin, 0)
        losses = F.relu(d_ap - d_an + self.margin)
        loss = losses.mean()

        # 4. Diagnostic Metrics
        active_triplets = (losses > 0.0).float().mean().item()
        mean_pos = d_ap.mean().item()
        mean_neg = d_an.mean().item()
        separation_gap = mean_neg - mean_pos

        metrics = {
            "loss": loss.item(),
            "mean_pos_dist": mean_pos,
            "mean_neg_dist": mean_neg,
            "separation_gap": separation_gap,
            "active_triplets_ratio": active_triplets,
        }

        return loss, metrics

