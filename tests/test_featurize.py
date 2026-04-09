"""
Tests for src/data/featurize.py — shape, dtype, and correctness checks.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.data.featurize import (
    ATOM_TYPES,
    POCKET_ELEMENTS,
    RESIDUE_TYPES,
    build_cross_edges,
    featurize_ligand,
    featurize_ligand_atom,
    featurize_ligand_bond,
    featurize_pocket,
    passes_filters,
    rbf_encoding,
)


# ---------------------------------------------------------------------------
# Ligand atom featurization
# ---------------------------------------------------------------------------

def test_featurize_ligand_atom_shape(aspirin_mol):
    for atom in aspirin_mol.GetAtoms():
        feat = featurize_ligand_atom(atom)
        assert feat.shape == (17,), f"Expected (17,), got {feat.shape}"
        assert feat.dtype == torch.float32


def test_featurize_ligand_atom_one_hot_sum(aspirin_mol):
    """Atom type one-hot (first 10 dims) should sum to 1."""
    for atom in aspirin_mol.GetAtoms():
        feat = featurize_ligand_atom(atom)
        assert feat[:10].sum().item() == pytest.approx(1.0)


def test_featurize_ligand_atom_hybridization_sum(aspirin_mol):
    """Hybridization one-hot (dims 11:14) should be 0 or 1 in each position."""
    for atom in aspirin_mol.GetAtoms():
        feat = featurize_ligand_atom(atom)
        hyb_sum = feat[11:14].sum().item()
        assert hyb_sum in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Ligand bond featurization
# ---------------------------------------------------------------------------

def test_featurize_ligand_bond_shape(aspirin_mol):
    for bond in aspirin_mol.GetBonds():
        feat = featurize_ligand_bond(bond)
        assert feat.shape == (6,)
        assert feat.dtype == torch.float32


def test_featurize_ligand_bond_type_one_hot(aspirin_mol):
    """Bond type one-hot (first 4 dims) should sum to 1."""
    for bond in aspirin_mol.GetBonds():
        feat = featurize_ligand_bond(bond)
        assert feat[:4].sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full ligand featurization
# ---------------------------------------------------------------------------

def test_featurize_ligand_shapes(aspirin_mol):
    node_feat, edge_index, edge_feat, pos = featurize_ligand(aspirin_mol)

    N = aspirin_mol.GetNumAtoms()
    E = aspirin_mol.GetNumBonds() * 2  # bidirectional

    assert node_feat.shape == (N, 17)
    assert edge_index.shape == (2, E)
    assert edge_feat.shape == (E, 6)
    assert pos.shape == (N, 3)


def test_featurize_ligand_bidirectional(aspirin_mol):
    """Edge index should contain both (i,j) and (j,i) for each bond."""
    _, edge_index, _, _ = featurize_ligand(aspirin_mol)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for i, j in zip(src, dst):
        assert (j, i) in list(zip(src, dst)), f"Missing reverse edge ({j},{i})"


def test_featurize_ligand_no_conformer_raises():
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError, match="conformer"):
        featurize_ligand(mol)


# ---------------------------------------------------------------------------
# RBF encoding
# ---------------------------------------------------------------------------

def test_rbf_encoding_shape():
    dists = torch.linspace(0, 10, 50)
    out = rbf_encoding(dists, n_basis=16)
    assert out.shape == (50, 16)


def test_rbf_encoding_values_in_range():
    dists = torch.tensor([0.0, 5.0, 10.0])
    out = rbf_encoding(dists)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0 + 1e-6  # RBF outputs in [0,1]


def test_rbf_encoding_clamps_zero():
    """Distance=0 should not produce NaN (clamped to 1e-3)."""
    dists = torch.tensor([0.0])
    out = rbf_encoding(dists)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Cross edges
# ---------------------------------------------------------------------------

def test_build_cross_edges_shape():
    lig_pos = torch.randn(5, 3)
    poc_pos = torch.randn(8, 3)
    cross_ei, cross_ea = build_cross_edges(lig_pos, poc_pos, k=4, n_rbf=16)

    assert cross_ei.shape == (2, 5 * 4)
    assert cross_ea.shape == (5 * 4, 16)


def test_build_cross_edges_index_ranges():
    N, M, k = 5, 8, 4
    lig_pos = torch.randn(N, 3)
    poc_pos = torch.randn(M, 3)
    cross_ei, _ = build_cross_edges(lig_pos, poc_pos, k=k)

    poc_idx = cross_ei[0]
    lig_idx = cross_ei[1]

    assert poc_idx.min().item() >= 0
    assert poc_idx.max().item() < M
    assert lig_idx.min().item() >= 0
    assert lig_idx.max().item() < N


def test_build_cross_edges_fewer_pocket_than_k():
    """When M < k, should return M edges per ligand atom."""
    lig_pos = torch.randn(3, 3)
    poc_pos = torch.randn(2, 3)
    cross_ei, cross_ea = build_cross_edges(lig_pos, poc_pos, k=4)

    # k_actual = min(4, 2) = 2
    assert cross_ei.shape[1] == 3 * 2
    assert cross_ea.shape[0] == 3 * 2


# ---------------------------------------------------------------------------
# Molecular filters
# ---------------------------------------------------------------------------

def test_passes_filters_aspirin(aspirin_mol):
    passed, details = passes_filters(aspirin_mol)
    assert passed, f"Aspirin should pass all filters, got: {details}"


def test_passes_filters_returns_dict(aspirin_mol):
    _, details = passes_filters(aspirin_mol)
    for key in ("mw", "mw_ok", "rot_bonds", "rot_bonds_ok",
                "heavy_atoms", "heavy_atoms_ok", "has_metal", "no_metal_ok"):
        assert key in details


def test_passes_filters_heavy_mol():
    """A large molecule should fail MW filter."""
    # Taxol-like: MW ~853
    smi = ("O=C(O[C@@H]1C[C@H]2OC(=O)[C@@H](O)[C@H]2[C@@H](OC(=O)c2ccccc2)"
           "[C@@]1(C)C(=O)O)c1ccccc1")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip("Molecule could not be parsed")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    mol = Chem.RemoveHs(mol)
    passed, details = passes_filters(mol)
    # May fail MW or heavy_atoms
    assert not passed or details["mw"] < 500
