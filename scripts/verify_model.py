"""
ResNet-50 Metric Architecture Verification Script.

Tests:
1. Backbone parameter freezing and trainable parameter statistics.
2. Single-image and Triplet forward pass tensor shapes: [B, 512].
3. L2 Normalization correctness: ||z||_2 == 1.0 ± 1e-5.
4. Deterministic evaluation test: identical image yields identical embedding (cosine distance ≈ 0).
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from src.config import load_config
from src.models.siamese_net import ResNet50MetricNet


def verify_model_architecture(config_path: str = "configs/config.yaml") -> bool:
    cfg = load_config(config_path)
    embedding_dim = cfg["model"]["embedding_dim"]
    freeze_early = cfg["model"]["freeze_early_layers"]
    batch_size = 4

    print("=" * 60)
    print("STEP 1: Instantiating ResNet50MetricNet (Pretrained Backbone)")
    print("=" * 60)
    model = ResNet50MetricNet(
        embedding_dim=embedding_dim,
        pretrained=True,
        freeze_early_layers=freeze_early,
    )
    model.eval()

    # Parameter Analysis
    param_summary = model.get_parameter_summary()
    print("\nPARAMETER ANALYSIS:")
    print(f"  • Total Parameters:     {param_summary['total_parameters']:,}")
    print(f"  • Trainable Parameters: {param_summary['trainable_parameters']:,}")
    print(f"  • Frozen Parameters:    {param_summary['frozen_parameters']:,}")
    print(f"  • Trainable Percentage: {param_summary['trainable_percentage']}%")

    assert param_summary["frozen_parameters"] > 0, "Error: Early layers were not frozen!"
    assert param_summary["trainable_parameters"] > 0, "Error: No trainable parameters found!"
    print("  ✓ Layer freezing policy verified.")

    print("\n" + "=" * 60)
    print("STEP 2: Testing Single-Image Forward Pass (Inference Mode)")
    print("=" * 60)
    dummy_x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        emb = model(dummy_x)

    print(f"  • Input Shape:  {dummy_x.shape}")
    print(f"  • Output Shape: {emb.shape} (Expected: [{batch_size}, {embedding_dim}])")
    assert emb.shape == (batch_size, embedding_dim), f"Unexpected embedding shape: {emb.shape}"
    print("  ✓ Single forward pass dimensions verified.")

    print("\n" + "=" * 60)
    print("STEP 3: Testing Triplet Forward Pass (Training Mode)")
    print("=" * 60)
    dummy_a = torch.randn(batch_size, 3, 224, 224)
    dummy_p = torch.randn(batch_size, 3, 224, 224)
    dummy_n = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        a_emb, p_emb, n_emb = model.forward_triplet(dummy_a, dummy_p, dummy_n)

    print(f"  • Anchor Embedding Shape:   {a_emb.shape}")
    print(f"  • Positive Embedding Shape: {p_emb.shape}")
    print(f"  • Negative Embedding Shape: {n_emb.shape}")

    assert a_emb.shape == (batch_size, embedding_dim)
    assert p_emb.shape == (batch_size, embedding_dim)
    assert n_emb.shape == (batch_size, embedding_dim)
    print("  ✓ Triplet forward pass dimensions verified.")

    print("\n" + "=" * 60)
    print("STEP 4: Testing L2 Normalization (Unit Hypersphere Projection)")
    print("=" * 60)
    # Compute L2 norms along dimension 1
    norms_a = torch.norm(a_emb, p=2, dim=1)
    norms_p = torch.norm(p_emb, p=2, dim=1)
    norms_n = torch.norm(n_emb, p=2, dim=1)

    print(f"  • Anchor L2 Norms:   {norms_a.tolist()}")
    print(f"  • Positive L2 Norms: {norms_p.tolist()}")
    print(f"  • Negative L2 Norms: {norms_n.tolist()}")

    # Check that all norms are 1.0 within numerical float32 epsilon (1e-5)
    assert torch.allclose(norms_a, torch.ones_like(norms_a), atol=1e-5), "Anchor norm != 1.0"
    assert torch.allclose(norms_p, torch.ones_like(norms_p), atol=1e-5), "Positive norm != 1.0"
    assert torch.allclose(norms_n, torch.ones_like(norms_n), atol=1e-5), "Negative norm != 1.0"
    print("  ✓ L2 Normalization strictly verified: all vectors lie on unit hypersphere.")

    print("\n" + "=" * 60)
    print("STEP 5: Testing Deterministic Inference & Metric Consistency")
    print("=" * 60)
    # Identical input must produce identical embedding in eval mode
    with torch.no_grad():
        emb_1 = model(dummy_x[:1])
        emb_2 = model(dummy_x[:1])

    cos_sim = F.cosine_similarity(emb_1, emb_2).item()
    euclidean_dist = torch.dist(emb_1, emb_2).item()

    print(f"  • Cosine Similarity (Identical Inputs): {cos_sim:.7f} (Expected: 1.0000000)")
    print(f"  • Euclidean Distance (Identical Inputs): {euclidean_dist:.7f} (Expected: 0.0000000)")

    assert abs(cos_sim - 1.0) < 1e-5, f"Cosine similarity deviated: {cos_sim}"
    assert euclidean_dist < 1e-5, f"Euclidean distance deviated: {euclidean_dist}"
    print("  ✓ Deterministic inference and metric consistency strictly verified.")

    print("\n" + "=" * 60)
    print("ALL PHASE 2 MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    verify_model_architecture()

