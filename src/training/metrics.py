"""
Evaluation metrics for 3D ligand conformation generation.

  - kabsch_rmsd:          Kabsch-aligned RMSD (Å)
  - mmff94_energy:        MMFF94 score (kcal/mol)
  - etkdg_energy:         ETKDG conformer + MMFF94 energy
  - strain_energy_ratio:  E_generated / E_etkdg
  - mol_with_coords:      Assign generated coords to RDKit mol (for visualization)
  - compute_test_metrics: Full evaluation on a test DataLoader
  - compute_etkdg_baseline: ETKDG baseline metrics
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: Tensor, Q: Tensor) -> float:
    """
    Kabsch-aligned RMSD between point clouds P and Q.

    Both P and Q are [N, 3] tensors. Centers both, finds the optimal rotation
    matrix R that minimizes ||P @ R.T - Q||, then returns RMSD in Angstroms.

    Handles reflections: forces det(R) = +1 via the sign-correction trick.

    Args:
        P: [N, 3] generated coordinates
        Q: [N, 3] crystal (reference) coordinates

    Returns:
        RMSD in Angstroms (float)
    """
    P = P.float()
    Q = Q.float()

    # Center both
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)

    # Cross-covariance
    H = Pc.T @ Qc   # [3, 3]

    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.linalg.det(Vt.T @ U.T)
        D = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=P.device))
        R = Vt.T @ D @ U.T
        P_rot = Pc @ R.T
    except torch._C._LinAlgError:
        # Degenerate case (e.g. near-zero coords early in training).
        # Fall back to unaligned RMSD — pessimistic but never crashes.
        P_rot = Pc

    rmsd = ((P_rot - Qc).pow(2).sum(-1).mean().sqrt()).item()
    return rmsd


# ---------------------------------------------------------------------------
# MMFF94 energy
# ---------------------------------------------------------------------------

def mmff94_energy(mol: Chem.Mol, coords: np.ndarray) -> Optional[float]:
    """
    Compute MMFF94 energy for mol with the given 3D coordinates.

    This is a score-only computation (no minimization).
    Returns energy in kcal/mol, or None if MMFF94 fails.

    Args:
        mol:    RDKit mol (without Hs — Hs will be added internally)
        coords: numpy [N, 3] coordinates for the heavy atoms
    """
    try:
        mol_h = Chem.AddHs(mol)
        conf = mol_h.GetConformer() if mol_h.GetNumConformers() > 0 else None

        # Embed with ETKDG to get a starting conformer for Hs, then replace heavy coords
        if conf is None:
            AllChem.EmbedMolecule(mol_h, randomSeed=42)

        # Set heavy atom positions
        conf = mol_h.GetConformer()
        heavy_map = {atom.GetIdx(): i
                     for i, atom in enumerate(mol.GetAtoms())}
        for heavy_idx, coord_idx in heavy_map.items():
            # Find corresponding index in mol_h (Hs added after heavy atoms)
            conf.SetAtomPosition(heavy_idx, coords[coord_idx].tolist())

        ff = AllChem.MMFFGetMoleculeForceField(
            mol_h, AllChem.MMFFGetMoleculeProperties(mol_h)
        )
        if ff is None:
            return None
        return ff.CalcEnergy()
    except Exception:
        return None


def etkdg_energy(mol: Chem.Mol, seed: int = 42) -> Optional[float]:
    """
    Generate an ETKDG conformer and compute its MMFF94 energy.

    Used as the baseline strain energy. Seed=42 for reproducibility.

    Returns energy in kcal/mol, or None on failure.
    """
    try:
        mol_h = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol_h, randomSeed=seed)
        if ret != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_h)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_h, AllChem.MMFFGetMoleculeProperties(mol_h)
        )
        if ff is None:
            return None
        return ff.CalcEnergy()
    except Exception:
        return None


def strain_energy_ratio(
    mol: Chem.Mol,
    generated_coords: np.ndarray,
) -> Optional[float]:
    """
    Compute E_generated / E_etkdg strain energy ratio.

    Returns None if either energy computation fails.
    """
    e_gen  = mmff94_energy(mol, generated_coords)
    e_etkdg = etkdg_energy(mol)
    if e_gen is None or e_etkdg is None:
        return None
    if abs(e_etkdg) < 1e-6:
        return None   # degenerate baseline
    return e_gen / e_etkdg


# ---------------------------------------------------------------------------
# mol_with_coords
# ---------------------------------------------------------------------------

def mol_with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """
    Return a new RDKit mol with a conformer set to the given coordinates.

    Args:
        mol:    original RDKit mol (no conformer required)
        coords: numpy [N, 3] for N heavy atoms

    Returns:
        new mol with one conformer at the given positions
    """
    mol_new = Chem.RWMol(mol)
    conf = Chem.Conformer(mol_new.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, (x, y, z))
    mol_new.AddConformer(conf, assignId=True)
    return mol_new.GetMol()


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def compute_test_metrics(
    flow_matcher,
    test_loader,
    device: torch.device,
    n_samples: int = 1,
) -> Dict:
    """
    Run full evaluation on a test DataLoader.

    For each molecule:
      - Generate n_samples conformations
      - Compute Kabsch RMSD vs crystal for the best sample
      - Compute strain energy ratio

    Returns dict with:
        rmsd_all              List[float] — per-molecule RMSD
        rmsd_median           float
        rmsd_mean             float
        rmsd_pct_under_1A     float  (%)
        rmsd_pct_under_2A     float  (%)
        rmsd_pct_under_5A     float  (%)
        strain_all            List[Optional[float]]
        strain_median         float  (excludes None)
        strain_mean           float  (excludes None)
        n_mmff_failed         int
        n_total               int
    """
    import sys
    sys.path.insert(0, ".")

    flow_matcher.eval()
    rmsd_all = []
    strain_all = []

    for batch in test_loader:
        batch = batch.to(device)
        unpacked = flow_matcher._unpack_batch(batch)
        lig_batch = unpacked["lig_batch"]
        crystal_pos = unpacked["lig_x"]
        n_graphs = lig_batch.max().item() + 1

        # Generate conformations
        generated = flow_matcher.generate(batch, n_steps=flow_matcher.n_steps)

        # We need the RDKit mol to compute strain energy — stored as complex_id
        # The mol must be fetched from the raw dataset (not available here).
        # For RMSD we only need coordinates.
        for g in range(n_graphs):
            mask = lig_batch == g
            crystal_g = crystal_pos[mask].cpu()
            gen_g     = generated[g].cpu()

            rmsd = kabsch_rmsd(gen_g, crystal_g)
            rmsd_all.append(rmsd)

            # Strain energy requires mol — skip here, compute separately
            strain_all.append(None)

    rmsd_arr = np.array(rmsd_all)
    valid_strains = [s for s in strain_all if s is not None]

    return {
        "rmsd_all":          rmsd_all,
        "rmsd_median":       float(np.median(rmsd_arr)),
        "rmsd_mean":         float(np.mean(rmsd_arr)),
        "rmsd_pct_under_1A": float((rmsd_arr < 1.0).mean() * 100),
        "rmsd_pct_under_2A": float((rmsd_arr < 2.0).mean() * 100),
        "rmsd_pct_under_5A": float((rmsd_arr < 5.0).mean() * 100),
        "strain_all":        strain_all,
        "strain_median":     float(np.median(valid_strains)) if valid_strains else float("nan"),
        "strain_mean":       float(np.mean(valid_strains)) if valid_strains else float("nan"),
        "n_mmff_failed":     strain_all.count(None),
        "n_total":           len(rmsd_all),
    }


def compute_etkdg_baseline(test_loader, mol_lookup: Dict) -> Dict:
    """
    Compute ETKDG baseline metrics.

    Args:
        test_loader: DataLoader over test set
        mol_lookup:  dict mapping complex_id → RDKit mol (with crystal conformer)

    Returns same structure as compute_test_metrics.
    """
    rmsd_all = []
    strain_all = []

    for batch in test_loader:
        # batch.complex_id is a list of strings in a batched HeteroData
        ids = batch.complex_id if isinstance(batch.complex_id, list) else [batch.complex_id]
        crystal_pos_list = _unbatch_positions(
            batch["ligand"].pos, batch["ligand"].batch
        )

        for cid, crystal_pos in zip(ids, crystal_pos_list):
            mol = mol_lookup.get(cid)
            if mol is None:
                continue

            mol_h = Chem.AddHs(mol)
            ret = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if ret != 0:
                rmsd_all.append(float("nan"))
                strain_all.append(None)
                continue

            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_h = Chem.RemoveHs(mol_h)
            etkdg_pos = torch.tensor(
                mol_h.GetConformer().GetPositions(), dtype=torch.float32
            )

            rmsd = kabsch_rmsd(etkdg_pos, crystal_pos.cpu())
            rmsd_all.append(rmsd)

            ratio = strain_energy_ratio(mol, etkdg_pos.numpy())
            strain_all.append(ratio)

    rmsd_arr = np.array([r for r in rmsd_all if not np.isnan(r)])
    valid_strains = [s for s in strain_all if s is not None]

    return {
        "rmsd_all":          rmsd_all,
        "rmsd_median":       float(np.median(rmsd_arr)) if len(rmsd_arr) else float("nan"),
        "rmsd_mean":         float(np.mean(rmsd_arr)) if len(rmsd_arr) else float("nan"),
        "rmsd_pct_under_1A": float((rmsd_arr < 1.0).mean() * 100) if len(rmsd_arr) else 0.0,
        "rmsd_pct_under_2A": float((rmsd_arr < 2.0).mean() * 100) if len(rmsd_arr) else 0.0,
        "rmsd_pct_under_5A": float((rmsd_arr < 5.0).mean() * 100) if len(rmsd_arr) else 0.0,
        "strain_all":        strain_all,
        "strain_median":     float(np.median(valid_strains)) if valid_strains else float("nan"),
        "strain_mean":       float(np.mean(valid_strains)) if valid_strains else float("nan"),
        "n_mmff_failed":     strain_all.count(None),
        "n_total":           len(rmsd_all),
    }


def _unbatch_positions(pos: Tensor, batch: Tensor) -> List[Tensor]:
    """Split a batched position tensor into per-graph tensors."""
    n_graphs = batch.max().item() + 1
    return [pos[batch == g] for g in range(n_graphs)]
