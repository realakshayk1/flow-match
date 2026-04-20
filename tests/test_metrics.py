import pytest
import numpy as np
import torch
from rdkit import Chem
from src.training.metrics import (
    kabsch_rmsd, 
    strain_energy_ratio, 
    compute_etkdg_baseline, 
    mmff94_energy, 
    mol_with_coords
)
from torch_geometric.data import HeteroData

def test_kabsch_requires_equal_atom_count():
    # 16 vs 18 atom count check
    P = torch.randn(16, 3)
    Q = torch.randn(18, 3)
    rmsd = kabsch_rmsd(P, Q)
    assert np.isnan(rmsd), "Expected NaN when shapes mismatch instead of crash."

def test_etkdg_baseline_skips_atom_mismatch():
    # Mock data with mismatched atom counts (mol has 16 heavy atoms, layout has 18)
    mol = Chem.MolFromSmiles("C" * 16)
    
    data = HeteroData()
    data.complex_id = "test_1"
    data["ligand"].pos = torch.randn(18, 3)
    data["ligand"].batch = torch.zeros(18, dtype=torch.long)
    
    mol_lookup = {"test_1": mol}
    
    metrics = compute_etkdg_baseline([data], mol_lookup)
    
    assert metrics["failures"]["n_atom_mismatch"] == 1
    assert metrics["failures"]["n_success"] == 0

def test_strain_metric_returns_finite_for_crystal_pose():
    # Benzene molecule
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol_h = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol_h, randomSeed=42)
    Chem.AllChem.MMFFOptimizeMolecule(mol_h)
    mol_h = Chem.RemoveHs(mol_h)
    
    pos = mol_h.GetConformer().GetPositions()
    
    # Evaluate strain
    ratio, reason = strain_energy_ratio(mol, pos)
    
    assert reason is None
    assert ratio is not None
    assert np.isfinite(ratio)
    
def test_strain_metric_invalid_geometry_check():
    # Provide explicitly corrupted coords (clashing)
    mol = Chem.MolFromSmiles("CC")
    # Distance between atoms = 0 < 0.4 criteria
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ratio, reason = strain_energy_ratio(mol, pos)
    
    assert ratio is None
    assert reason is not None
    assert "invalid_pose_geometry: severe_clash" in reason

def test_conformer_assignment_preserves_atom_count():
    mol = Chem.MolFromSmiles("CCO")
    pos = np.zeros((mol.GetNumAtoms(), 3))
    new_mol = mol_with_coords(mol, pos)
    
    assert new_mol.GetNumAtoms() == mol.GetNumAtoms()
    assert new_mol.GetNumConformers() == 1
    assert new_mol.GetConformer().GetPositions().shape[0] == 3
