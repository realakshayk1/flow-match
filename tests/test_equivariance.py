"""
Equivariance unit tests for EGNNFlowModel.
HARD GATE: All tests must pass with max error < 1e-4 before training begins.

Tests:
  1. test_rotation_equivariance    — v(R·x) = R·v(x)
  2. test_translation_invariance   — v(x + c) = v(x)
  3. test_reflection_equivariance  — v(S·x) = S·v(x) for improper rotation S
  4. test_output_shape             — v has shape [N_lig, 3]
  5. test_param_count              — model has < 500K parameters
"""

import pytest
import torch

from src.models.egnn import EGNNFlowModel, count_parameters

EQUIVARIANCE_TOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_rotation(device="cpu") -> torch.Tensor:
    """Generate a random proper rotation matrix (det = +1)."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] = Q[:, 0] * -1
    return Q  # [3, 3]


def _run_model(model: EGNNFlowModel, batch: dict) -> torch.Tensor:
    return model(
        lig_x=batch["lig_x"],
        lig_h=batch["lig_h"],
        poc_x=batch["poc_x"],
        poc_h=batch["poc_h"],
        lig_edge_index=batch["lig_edge_index"],
        lig_edge_attr=batch["lig_edge_attr"],
        poc_edge_index=batch["poc_edge_index"],
        poc_edge_attr=batch["poc_edge_attr"],
        cross_edge_index=batch["cross_edge_index"],
        cross_edge_attr=batch["cross_edge_attr"],
        t=batch["t"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(tiny_egnn_batch):
    """Output velocity must have shape [N_lig, 3]."""
    model = EGNNFlowModel()
    model.eval()
    with torch.no_grad():
        v = _run_model(model, tiny_egnn_batch)
    assert v.shape == (5, 3), f"Expected [5, 3], got {v.shape}"


def test_param_count():
    """Model must have fewer than 500K trainable parameters."""
    model = EGNNFlowModel()
    n_params = count_parameters(model)
    assert n_params < 500_000, (
        f"Model has {n_params:,} parameters — exceeds 500K limit. "
        "Reduce hidden_dim or n_layers."
    )
    print(f"\nModel parameters: {n_params:,}")


def test_rotation_equivariance(tiny_egnn_batch):
    """
    SE(3) rotation equivariance:
        v(R · x) = R · v(x)
    for any proper rotation matrix R.
    """
    model = EGNNFlowModel()
    model.eval()

    R = _random_rotation()

    with torch.no_grad():
        v1 = _run_model(model, tiny_egnn_batch)  # v(x)

        rotated_batch = {
            **tiny_egnn_batch,
            "lig_x": tiny_egnn_batch["lig_x"] @ R.T,  # R · x (row-vector convention)
            "poc_x": tiny_egnn_batch["poc_x"] @ R.T,
        }
        v2 = _run_model(model, rotated_batch)    # v(R·x)

    expected = v1 @ R.T   # R · v(x)
    err = (v2 - expected).abs().max().item()

    assert err < EQUIVARIANCE_TOL, (
        f"Rotation equivariance FAILED: max error = {err:.2e} "
        f"(tolerance = {EQUIVARIANCE_TOL:.0e})\n"
        "Check: edge direction (src→dst), scatter_add_ target, fixed_mask."
    )


def test_translation_invariance(tiny_egnn_batch):
    """
    SE(3) translation invariance:
        v(x + c) = v(x)
    for any constant translation vector c.

    Velocity is a vector (not position), so it must be translation-invariant.
    """
    model = EGNNFlowModel()
    model.eval()

    c = torch.tensor([3.0, -2.0, 1.5])  # arbitrary translation

    with torch.no_grad():
        v1 = _run_model(model, tiny_egnn_batch)

        shifted_batch = {
            **tiny_egnn_batch,
            "lig_x": tiny_egnn_batch["lig_x"] + c,
            "poc_x": tiny_egnn_batch["poc_x"] + c,
        }
        v2 = _run_model(model, shifted_batch)

    err = (v2 - v1).abs().max().item()

    assert err < EQUIVARIANCE_TOL, (
        f"Translation invariance FAILED: max error = {err:.2e} "
        f"(tolerance = {EQUIVARIANCE_TOL:.0e})\n"
        "Check: coordinates are used only as differences (x_i - x_j), never raw."
    )


def test_reflection_equivariance(tiny_egnn_batch):
    """
    Improper rotation (reflection) equivariance:
        v(S · x) = S · v(x)
    where S is a reflection matrix (det = -1).

    Tests the sign of phi_x — failure here typically indicates an issue
    with the coordinate update formula or the message computation.
    """
    model = EGNNFlowModel()
    model.eval()

    # Reflection through the XY plane: negate z-axis
    S = torch.diag(torch.tensor([1.0, 1.0, -1.0]))

    with torch.no_grad():
        v1 = _run_model(model, tiny_egnn_batch)

        reflected_batch = {
            **tiny_egnn_batch,
            "lig_x": tiny_egnn_batch["lig_x"] @ S,
            "poc_x": tiny_egnn_batch["poc_x"] @ S,
        }
        v2 = _run_model(model, reflected_batch)

    expected = v1 @ S   # S · v(x)
    err = (v2 - expected).abs().max().item()

    assert err < EQUIVARIANCE_TOL, (
        f"Reflection equivariance FAILED: max error = {err:.2e} "
        f"(tolerance = {EQUIVARIANCE_TOL:.0e})\n"
        "Check: phi_x has no final activation (must allow negative weights)."
    )


def test_equivariance_multiple_rotations(tiny_egnn_batch):
    """Run rotation equivariance test with 5 different random rotations."""
    model = EGNNFlowModel()
    model.eval()

    torch.manual_seed(99)
    for i in range(5):
        R = _random_rotation()
        with torch.no_grad():
            v1 = _run_model(model, tiny_egnn_batch)
            rotated_batch = {
                **tiny_egnn_batch,
                "lig_x": tiny_egnn_batch["lig_x"] @ R.T,
                "poc_x": tiny_egnn_batch["poc_x"] @ R.T,
            }
            v2 = _run_model(model, rotated_batch)

        expected = v1 @ R.T
        err = (v2 - expected).abs().max().item()
        assert err < EQUIVARIANCE_TOL, (
            f"Rotation {i+1}/5 equivariance FAILED: max error = {err:.2e}"
        )
