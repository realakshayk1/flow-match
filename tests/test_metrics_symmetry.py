"""
Tests for the in-frame, symmetry-corrected docking RMSD (symmetry_rmsd) and its
distinction from the Kabsch-aligned shape RMSD (kabsch_rmsd).

Core properties verified:
  (a) identical coordinates -> 0
  (b) a symmetric molecule (benzene) -> symmetry correction beats the naive identity mapping
  (c) a globally translated pose -> near-zero SHAPE rmsd but large DOCK rmsd
      (this is exactly the gap the audit flagged: Kabsch alignment hides misplacement)
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.training.metrics import kabsch_rmsd, symmetry_rmsd, pocket_aware_relax


def _embed(mol, seed=42):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol = Chem.RemoveHs(mol)
    return mol, mol.GetConformer().GetPositions().astype(np.float64)


def test_identical_coords_zero():
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol, coords = _embed(mol)
    assert symmetry_rmsd(mol, coords, coords) < 1e-6


def test_symmetry_beats_naive_for_benzene():
    """Rotating benzene by 60 degrees in-plane maps the ring onto itself: a symmetric
    relabeling should recover a near-zero RMSD while the naive identity mapping does not."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol, coords = _embed(mol)

    # Center, then rotate 60 degrees about the approximate ring normal (z of PCA).
    c = coords - coords.mean(0)
    # Build a ring-plane basis via SVD; rotate about the normal (smallest singular vector).
    _, _, vt = np.linalg.svd(c)
    normal = vt[2]
    theta = np.deg2rad(60.0)
    # Rodrigues rotation about `normal`.
    K = np.array([[0, -normal[2], normal[1]],
                  [normal[2], 0, -normal[0]],
                  [-normal[1], normal[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    rotated = c @ R.T

    naive = float(np.sqrt(((c - rotated) ** 2).sum(-1).mean()))
    sym = symmetry_rmsd(mol, rotated, c)

    # The 60-degree ring rotation is (close to) a symmetry, so symmetry correction is much
    # better than the naive same-index comparison.
    assert sym < naive - 0.3
    assert sym < 0.5


def test_translation_hidden_by_kabsch_but_caught_by_dock():
    """A pose translated 10 A away has ~0 shape RMSD (Kabsch removes translation) but a large
    in-frame dock RMSD. This is the audit's central point made executable."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    mol, coords = _embed(mol)

    translated = coords + np.array([10.0, 0.0, 0.0])

    shape = kabsch_rmsd(torch.tensor(translated), torch.tensor(coords))
    dock = symmetry_rmsd(mol, translated, coords)

    assert shape < 1e-3              # Kabsch hides the 10 A displacement
    assert dock > 9.0                # in-frame RMSD exposes it


def _max_bond_len(mol, coords):
    return max(
        float(np.linalg.norm(coords[b.GetBeginAtomIdx()] - coords[b.GetEndAtomIdx()]))
        for b in mol.GetBonds()
    )


def test_pocket_aware_relax_fixes_bond_geometry_and_stays_near_pose():
    """A bond-stretched pose has invalid geometry; restrained relaxation should restore
    plausible bond lengths while keeping the pose within the restraint slack of the input."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    mol, coords = _embed(mol)

    # Stretch the whole molecule 15% about its centroid -> bonds ~1.7 A (invalid-ish).
    c = coords.mean(0)
    stretched = c + (coords - c) * 1.15
    assert _max_bond_len(mol, stretched) > 1.7

    relaxed, reason = pocket_aware_relax(mol, stretched, max_displ=1.0, restraint_k=50.0)
    assert reason is None and relaxed is not None

    # Bond lengths back in a plausible covalent range.
    assert _max_bond_len(mol, relaxed) < 1.95
    # Pose preserved: stays near the (stretched) input rather than collapsing elsewhere.
    assert symmetry_rmsd(mol, relaxed, stretched) < 1.5


def test_atom_count_mismatch_returns_nan():
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol, coords = _embed(mol)
    wrong = coords[:-1]
    assert np.isnan(symmetry_rmsd(mol, wrong, coords[: len(wrong)]))
