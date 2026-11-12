"""
Automated Unit Tests for Phase 2: ResNet-50 Metric Architecture.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.siamese_net import ResNet50MetricNet


@pytest.fixture(scope="module")
def model():
    """Instantiate model once for tests without downloading weights repeatedly."""
    net = ResNet50MetricNet(
        embedding_dim=512,
        pretrained=False,  # Use random weights for fast unit testing
        freeze_early_layers=True,
    )
    net.eval()
    return net


def test_parameter_freezing(model):
    """Verify that early layers are frozen and later layers are trainable."""
    # conv1 and layer1 parameters must have requires_grad == False
    for param in model.conv1.parameters():
        assert not param.requires_grad
    for param in model.layer1.parameters():
        assert not param.requires_grad
    for param in model.layer2.parameters():
        assert not param.requires_grad

    # layer3, layer4, and head must have requires_grad == True
    for param in model.layer3.parameters():
        assert param.requires_grad
    for param in model.layer4.parameters():
        assert param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_embedding_dimensions(model):
    """Test output shapes for single-image and triplet forward passes."""
    batch_size = 3
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, 512)

    with torch.no_grad():
        a_emb, p_emb, n_emb = model.forward_triplet(x, x, x)
    assert a_emb.shape == (batch_size, 512)
    assert p_emb.shape == (batch_size, 512)
    assert n_emb.shape == (batch_size, 512)


def test_l2_normalization_constraint(model):
    """Test that all output vectors lie on the unit hypersphere: ||z||_2 == 1.0."""
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    norms = torch.norm(out, p=2, dim=1)
    expected = torch.ones_like(norms)
    assert torch.allclose(norms, expected, atol=1e-5)


def test_deterministic_identical_similarity(model):
    """Test that identical images produce cosine similarity == 1.0 and euclidean distance == 0.0."""
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        emb1 = model(x)
        emb2 = model(x)
    cos_sim = F.cosine_similarity(emb1, emb2).item()
    euclidean = torch.dist(emb1, emb2).item()

    assert abs(cos_sim - 1.0) < 1e-5
    assert euclidean < 1e-5

