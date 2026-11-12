"""
Unit Tests for Phase 3: Triplet Margin Loss & Online Negative Mining.
"""

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import OnlineTripletLoss, pairwise_distance


def test_pairwise_distance_matrix():
    """Verify all-pairs Euclidean distance on unit vectors."""
    # Create orthogonal vectors in R^3
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0, 0.0]])
    vectors = torch.cat([v1, v2], dim=0)

    dist_matrix = pairwise_distance(vectors)
    assert dist_matrix.shape == (2, 2)
    # Self distance must be 0
    assert torch.isclose(dist_matrix[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(dist_matrix[1, 1], torch.tensor(0.0), atol=1e-5)
    # Orthogonal distance: sqrt(2 - 2 * 0) = sqrt(2) ≈ 1.4142
    expected_dist = torch.sqrt(torch.tensor(2.0))
    assert torch.isclose(dist_matrix[0, 1], expected_dist, atol=1e-4)


def test_online_triplet_loss_semi_hard():
    """Verify online triplet loss computation and active triplet detection."""
    criterion = OnlineTripletLoss(margin=0.2, semi_hard=True)

    # 4 samples, 2 classes: [0, 0, 1, 1]
    labels = torch.tensor([0, 0, 1, 1])

    # Case A: Well-separated embeddings (positives close, negatives far)
    # Anchor 0 and Pos 0 close to [1, 0, 0...]
    a_emb = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]]), p=2, dim=1)
    p_emb = F.normalize(torch.tensor([[0.99, 0.0], [0.99, 0.1], [0.0, 0.99], [0.1, 0.99]]), p=2, dim=1)

    loss_easy, metrics_easy = criterion(a_emb, p_emb, labels)
    assert loss_easy >= 0.0
    assert metrics_easy["mean_neg_dist"] > metrics_easy["mean_pos_dist"]

    # Case B: Confused embeddings (positives far, negatives close)
    # Anchor 0 far from Pos 0, but close to Negative
    p_emb_bad = F.normalize(torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]), p=2, dim=1)
    loss_hard, metrics_hard = criterion(a_emb, p_emb_bad, labels)

    # Hard loss should be significantly higher than easy loss
    assert loss_hard > loss_easy
    assert metrics_hard["active_triplets_ratio"] > 0.0


def test_checkpoint_roundtrip(tmp_path):
    """Test saving and loading model weights checkpoint."""
    from src.models.siamese_net import ResNet50MetricNet

    model_src = ResNet50MetricNet(embedding_dim=512, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    model_src.eval()
    with torch.no_grad():
        out_src = model_src(dummy_x)

    ckpt_file = tmp_path / "test_model.pt"
    torch.save({"model_state_dict": model_src.state_dict()}, ckpt_file)

    model_dst = ResNet50MetricNet(embedding_dim=512, pretrained=False)
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model_dst.load_state_dict(checkpoint["model_state_dict"])
    model_dst.eval()

    with torch.no_grad():
        out_dst = model_dst(dummy_x)

    # Weights must produce bit-for-bit identical embeddings
    assert torch.allclose(out_src, out_dst, atol=1e-6)

