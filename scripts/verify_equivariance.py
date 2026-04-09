"""
Standalone equivariance verification script (no pytest required).

Run after installing the conda environment:
    python scripts/verify_equivariance.py

Exits with code 0 if all checks pass, 1 if any fail.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.egnn import EGNNFlowModel, count_parameters

TOLERANCE = 1e-4


def make_test_batch():
    torch.manual_seed(42)
    return {
        "lig_h":    torch.randn(5, 17),
        "lig_x":    torch.randn(5, 3),
        "poc_h":    torch.randn(8, 29),
        "poc_x":    torch.randn(8, 3),
        "lig_edge_index": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                         [1, 0, 2, 1, 3, 2, 4, 3]]),
        "lig_edge_attr":  torch.randn(8, 6),
        "poc_edge_index": torch.zeros((2, 0), dtype=torch.long),
        "poc_edge_attr":  torch.zeros((0, 16)),
        "cross_edge_index": torch.stack([
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 4, 5, 2, 3, 6, 7]),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
        ]),
        "cross_edge_attr": torch.randn(20, 16),
        "t": torch.tensor(0.5),
    }


def run_model(model, batch):
    return model(**batch)


def random_rotation():
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] *= -1
    return Q


def check(name, passed, err=None):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if err is not None:
        msg += f"  (max error: {err:.2e})"
    print(msg)
    return passed


def main():
    print("=" * 50)
    print("EGNN Equivariance Verification")
    print("=" * 50)

    model = EGNNFlowModel()
    model.eval()

    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")

    results = []
    batch = make_test_batch()

    # 1. Output shape
    with torch.no_grad():
        v = run_model(model, batch)
    passed = v.shape == (5, 3)
    results.append(check("Output shape [5, 3]", passed))

    # 2. Parameter count
    passed = n_params < 500_000
    results.append(check(f"Params < 500K ({n_params:,})", passed))

    # 3. Rotation equivariance (5 rotations)
    print("\nRotation equivariance (5 random rotations):")
    all_rot_pass = True
    for i in range(5):
        torch.manual_seed(i * 100)
        R = random_rotation()
        with torch.no_grad():
            v1 = run_model(model, batch)
            rb = {**batch, "lig_x": batch["lig_x"] @ R.T, "poc_x": batch["poc_x"] @ R.T}
            v2 = run_model(model, rb)
        err = (v2 - v1 @ R.T).abs().max().item()
        passed = err < TOLERANCE
        all_rot_pass = all_rot_pass and passed
        results.append(check(f"  Rotation {i+1}/5", passed, err))

    # 4. Translation invariance
    print("\nTranslation invariance:")
    c = torch.tensor([3.0, -2.0, 1.5])
    with torch.no_grad():
        v1 = run_model(model, batch)
        sb = {**batch, "lig_x": batch["lig_x"] + c, "poc_x": batch["poc_x"] + c}
        v2 = run_model(model, sb)
    err = (v2 - v1).abs().max().item()
    passed = err < TOLERANCE
    results.append(check("Translation invariance", passed, err))

    # 5. Reflection equivariance
    print("\nReflection equivariance:")
    S = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    with torch.no_grad():
        v1 = run_model(model, batch)
        refb = {**batch, "lig_x": batch["lig_x"] @ S, "poc_x": batch["poc_x"] @ S}
        v2 = run_model(model, refb)
    err = (v2 - v1 @ S).abs().max().item()
    passed = err < TOLERANCE
    results.append(check("Reflection equivariance (XY plane)", passed, err))

    print("\n" + "=" * 50)
    n_pass = sum(results)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} checks passed")

    if all(results):
        print("ALL CHECKS PASSED — model is ready for training.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — do NOT proceed to training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
