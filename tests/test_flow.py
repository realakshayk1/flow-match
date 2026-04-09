"""
Tests for FlowMatcher — loss and inference shape/correctness.
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.egnn import EGNNFlowModel
from src.models.flow_model import FlowMatcher, _clamp_update


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hetero_batch(n_lig=5, n_poc=8, k=4):
    """Build a single-graph HeteroData for testing."""
    torch.manual_seed(0)
    data = HeteroData()

    data["ligand"].x   = torch.randn(n_lig, 17)
    data["ligand"].pos = torch.randn(n_lig, 3)
    data["ligand"].edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4],
         [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    data["ligand"].edge_attr = torch.randn(8, 6)
    data["ligand"].batch = torch.zeros(n_lig, dtype=torch.long)

    data["pocket"].x   = torch.randn(n_poc, 29)
    data["pocket"].pos = torch.randn(n_poc, 3)
    data["pocket"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["pocket"].edge_attr  = torch.zeros((0, 16))
    data["pocket"].batch = torch.zeros(n_poc, dtype=torch.long)

    # k nearest pocket atoms per ligand atom
    poc_idx = torch.arange(n_poc).repeat(n_lig)[:n_lig * k]
    lig_idx = torch.arange(n_lig).repeat_interleave(k)
    data["pocket", "to", "ligand"].edge_index = torch.stack([poc_idx, lig_idx])
    data["pocket", "to", "ligand"].edge_attr  = torch.randn(n_lig * k, 16)

    return data


@pytest.fixture
def single_graph_batch():
    return _make_hetero_batch()


@pytest.fixture
def flow_matcher():
    model = EGNNFlowModel()
    return FlowMatcher(model, n_steps=20)


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------

def test_compute_loss_is_scalar(flow_matcher, single_graph_batch):
    """compute_loss must return a scalar tensor."""
    loss = flow_matcher.compute_loss(single_graph_batch)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"


def test_compute_loss_is_positive(flow_matcher, single_graph_batch):
    """MSE loss must be non-negative."""
    loss = flow_matcher.compute_loss(single_graph_batch)
    assert loss.item() >= 0.0


def test_compute_loss_is_finite(flow_matcher, single_graph_batch):
    """Loss must not be NaN or inf."""
    loss = flow_matcher.compute_loss(single_graph_batch)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_compute_loss_backward(flow_matcher, single_graph_batch):
    """Loss must support backpropagation."""
    loss = flow_matcher.compute_loss(single_graph_batch)
    loss.backward()
    # Check that at least some parameters have gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in flow_matcher.parameters()
    )
    assert has_grad, "No parameter received a gradient"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_returns_list(flow_matcher, single_graph_batch):
    """generate() must return a list of tensors."""
    results = flow_matcher.generate(single_graph_batch)
    assert isinstance(results, list)


def test_generate_output_shape(flow_matcher, single_graph_batch):
    """Each generated tensor must have shape [N_lig_i, 3]."""
    results = flow_matcher.generate(single_graph_batch)
    assert len(results) == 1
    assert results[0].shape == (5, 3), f"Expected [5,3], got {results[0].shape}"


def test_generate_is_finite(flow_matcher, single_graph_batch):
    """Generated coordinates must be finite."""
    results = flow_matcher.generate(single_graph_batch)
    for r in results:
        assert torch.isfinite(r).all(), "Generated coordinates contain NaN or inf"


def test_generate_single(flow_matcher):
    """generate_single() convenience method produces correct shape."""
    torch.manual_seed(1)
    n_lig, n_poc, k = 6, 10, 4
    result = flow_matcher.generate_single(
        lig_h=torch.randn(n_lig, 17),
        poc_x=torch.randn(n_poc, 3),
        poc_h=torch.randn(n_poc, 29),
        lig_edge_index=torch.zeros((2, 0), dtype=torch.long),
        lig_edge_attr=torch.zeros((0, 6)),
        poc_edge_index=torch.zeros((2, 0), dtype=torch.long),
        poc_edge_attr=torch.zeros((0, 16)),
        cross_edge_index=torch.stack([
            torch.arange(n_poc).repeat(n_lig)[:n_lig * k],
            torch.arange(n_lig).repeat_interleave(k),
        ]),
        cross_edge_attr=torch.randn(n_lig * k, 16),
    )
    assert result.shape == (n_lig, 3)


def test_generate_different_each_call(flow_matcher, single_graph_batch):
    """Two generate() calls should produce different results (stochastic x_0)."""
    r1 = flow_matcher.generate(single_graph_batch)[0]
    r2 = flow_matcher.generate(single_graph_batch)[0]
    assert not torch.allclose(r1, r2), "Two generate() calls returned identical results"


# ---------------------------------------------------------------------------
# _clamp_update
# ---------------------------------------------------------------------------

def test_clamp_update_no_change_within_limit():
    """Small updates should not be clipped."""
    x_old = torch.zeros(5, 3)
    x_new = torch.ones(5, 3) * 0.1  # delta norm = sqrt(3) * 0.1 ≈ 0.17 < 5.0
    result = _clamp_update(x_new, x_old, max_delta=5.0)
    assert torch.allclose(result, x_new, atol=1e-6)


def test_clamp_update_clips_large_steps():
    """Large updates must be clipped to max_delta."""
    x_old = torch.zeros(1, 3)
    x_new = torch.tensor([[100.0, 0.0, 0.0]])
    result = _clamp_update(x_new, x_old, max_delta=5.0)
    delta_norm = (result - x_old).norm(dim=-1).item()
    assert abs(delta_norm - 5.0) < 1e-5, f"Expected delta=5.0, got {delta_norm:.4f}"
