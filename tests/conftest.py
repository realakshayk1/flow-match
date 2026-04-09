"""
Shared test fixtures.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture
def aspirin_mol():
    """RDKit mol for aspirin with an MMFF-optimized 3D conformer."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol


@pytest.fixture
def tiny_egnn_batch():
    """
    Minimal synthetic batch for EGNN tests.
    5 ligand atoms, 8 pocket atoms, pre-built edges.
    """
    torch.manual_seed(42)
    N_lig = 5
    N_poc = 8

    lig_h = torch.randn(N_lig, 17)
    lig_x = torch.randn(N_lig, 3)
    poc_h = torch.randn(N_poc, 29)
    poc_x = torch.randn(N_poc, 3)

    # Simple chain edges for ligand (bidirectional)
    lig_edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4],
         [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    lig_edge_attr = torch.randn(8, 6)

    # No intra-pocket edges for simplicity
    poc_edge_index = torch.zeros((2, 0), dtype=torch.long)
    poc_edge_attr = torch.zeros((0, 16), dtype=torch.float32)

    # Cross edges: each ligand atom connects to 4 pocket atoms
    poc_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 4, 5, 2, 3, 6, 7])
    lig_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    cross_edge_index = torch.stack([poc_idx, lig_idx], dim=0)
    cross_edge_attr = torch.randn(20, 16)

    t = torch.tensor(0.5)

    return {
        "lig_h": lig_h,
        "lig_x": lig_x,
        "poc_h": poc_h,
        "poc_x": poc_x,
        "lig_edge_index": lig_edges,
        "lig_edge_attr": lig_edge_attr,
        "poc_edge_index": poc_edge_index,
        "poc_edge_attr": poc_edge_attr,
        "cross_edge_index": cross_edge_index,
        "cross_edge_attr": cross_edge_attr,
        "t": t,
    }
