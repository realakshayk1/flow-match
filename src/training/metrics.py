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

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.distance import pdist

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
    if P.shape[0] != Q.shape[0]:
        return float("nan")

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
        # d.sign() == 0 (degenerate SVD) would make D singular → NaN RMSD;
        # treat as +1 (proper rotation) in that case.
        sign_d = d.sign() if d.abs() > 1e-8 else torch.ones(1, device=P.device).squeeze()
        D = torch.diag(torch.tensor([1.0, 1.0, sign_d.item()], device=P.device))
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

def mmff94_energy(mol: Chem.Mol, coords: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute MMFF94 energy for mol with the given 3D coordinates.

    This is a score-only computation (no minimization).
    Returns (energy in kcal/mol, failure_reason).
    """
    # 1. Reject invalid geometry
    if coords.shape[0] != mol.GetNumAtoms():
        return None, "invalid_pose_geometry: atom_count_mismatch"
    if not np.isfinite(coords).all():
        return None, "invalid_pose_geometry: non_finite_coords"
    if np.abs(coords).max() > 1000.0:
        return None, "invalid_pose_geometry: absurd_magnitude_coords"
    
    # Check for severe clashes
    if coords.shape[0] > 1:
        dists = pdist(coords)
        if dists.min() < 0.4:
            return None, "invalid_pose_geometry: severe_clash"

    try:
        mol_h = Chem.AddHs(mol)
        conf = mol_h.GetConformer() if mol_h.GetNumConformers() > 0 else None

        # Embed with ETKDG to get a starting conformer for Hs, then replace heavy coords
        if conf is None:
            ret = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if ret != 0:
                return None, "rdkit_mmff_setup_failure: embed_molecule_failed"

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
            return None, "rdkit_mmff_setup_failure: get_force_field_failed"
        
        energy = ff.CalcEnergy()
        if energy > 10000.0:
            return energy, "absurd_energy: >10000_kcal_mol"
        return energy, None
    except Exception as e:
        return None, f"rdkit_mmff_setup_failure: exception_{str(e)[:50]}"


def etkdg_energy(mol: Chem.Mol, seed: int = 42) -> Tuple[Optional[float], Optional[str]]:
    """
    Generate an ETKDG conformer and compute its MMFF94 energy.
    Returns (energy in kcal/mol, failure_reason)
    """
    try:
        mol_h = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol_h, randomSeed=seed)
        if ret != 0:
            return None, "rdkit_mmff_setup_failure: embed_molecule_failed"
        AllChem.MMFFOptimizeMolecule(mol_h)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_h, AllChem.MMFFGetMoleculeProperties(mol_h)
        )
        if ff is None:
            return None, "rdkit_mmff_setup_failure: get_force_field_failed"
        
        energy = ff.CalcEnergy()
        if energy > 10000.0:
            return energy, "absurd_energy: >10000_kcal_mol"
        return energy, None
    except Exception as e:
        return None, f"rdkit_mmff_setup_failure: exception_{str(e)[:50]}"


def strain_energy_ratio(
    mol: Chem.Mol,
    generated_coords: np.ndarray,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute E_generated / E_etkdg strain energy ratio.
    Returns (ratio, failure_reason).
    """
    e_gen, gen_reason  = mmff94_energy(mol, generated_coords)
    e_etkdg, etkdg_reason = etkdg_energy(mol)
    
    if e_gen is None:
        return None, f"gen_{gen_reason}"
    if e_etkdg is None:
        return None, f"etkdg_{etkdg_reason}"
    
    if abs(e_etkdg) < 1e-6:
        return None, "degenerate_baseline_energy_near_zero"
    
    ratio = e_gen / e_etkdg
    reason = None
    if gen_reason and "absurd_energy" in gen_reason:
        reason = "absurd_energy"
        
    return ratio, reason


# ---------------------------------------------------------------------------
# mol_with_coords
# ---------------------------------------------------------------------------

def mol_with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """
    Return a new RDKit mol with a conformer set to the given coordinates.
    """
    mol_new = Chem.RWMol(mol)
    conf = Chem.Conformer(mol_new.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, (x, y, z))
    mol_new.AddConformer(conf, assignId=True)
    return mol_new.GetMol()


def uff_minimize(mol: Chem.Mol, coords: np.ndarray, max_iters: int = 200) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Run RDKit UFF minimization starting from generated coordinates.
    Returns (minimized_heavy_atom_coords, failure_reason).
    Failure_reason is None on success.
    """
    if not np.isfinite(coords).all():
        return None, "non_finite_input_coords"
    if coords.shape[0] != mol.GetNumAtoms():
        return None, "atom_count_mismatch"

    try:
        mol_h = Chem.AddHs(mol)
        # Embed to place H atoms, then overwrite heavy atom positions
        ret = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if ret != 0:
            return None, "embed_failed"

        conf = mol_h.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, coords[i].tolist())

        result = AllChem.UFFOptimizeMolecule(mol_h, maxIters=max_iters)
        # result: 0 = converged, 1 = more iters needed, -1 = failure
        if result == -1:
            return None, "uff_optimization_failed"

        mol_noH = Chem.RemoveHs(mol_h)
        opt_coords = mol_noH.GetConformer().GetPositions()
        return opt_coords.astype(np.float32), None
    except Exception as e:
        return None, f"exception_{str(e)[:60]}"


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def compute_test_metrics(
    flow_matcher,
    test_loader,
    device: torch.device,
    n_samples: int = 1,
    debug_eval_examples: int = 0,
    dump_eval_predictions: str = "",
) -> Dict:
    """
    Run full evaluation on a test DataLoader.
    """
    import sys
    sys.path.insert(0, ".")

    import json
    import os

    flow_matcher.eval()
    rmsd_all = []
    strain_all = []
    strain_failures = {
        "invalid_pose_geometry": 0,
        "rdkit_mmff_setup_failure": 0,
        "absurd_energy": 0,
        "other": 0
    }
    
    debug_count = 0
    eval_dumps = []

    for batch in test_loader:
        batch = batch.to(device)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = int(lig_batch.max().item()) + 1

        # Generated conformations
        generated = flow_matcher.generate(batch, n_steps=flow_matcher.n_steps)

        # SMILES stored per-graph in HeteroData (added by preprocess.py).
        smiles_list = batch.ligand_meta_canonical_smiles if hasattr(batch, "ligand_meta_canonical_smiles") else (batch.smiles if hasattr(batch, "smiles") else None)
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        for g in range(n_graphs):
            mask  = lig_batch == g
            crystal_g = crystal_pos[mask].cpu()
            gen_g     = generated[g].cpu()

            rmsd = kabsch_rmsd(gen_g, crystal_g)
            rmsd_all.append(rmsd)

            complex_id = batch.complex_id[g] if hasattr(batch, "complex_id") and isinstance(batch.complex_id, list) else getattr(batch, "complex_id", f"batch_unk_{debug_count}")
            
            if debug_count < debug_eval_examples:
                coords_nan = torch.isnan(gen_g).any().item() or torch.isinf(gen_g).any().item()
                print(f"\n[DEBUG EVAL] Complex: {complex_id}")
                print(f"  Target shape: {crystal_g.shape} | Pred shape: {gen_g.shape}")
                print(f"  Pred domain: min={gen_g.min().item():.3f}, max={gen_g.max().item():.3f}, mean_norm={torch.norm(gen_g, dim=-1).mean().item():.3f}")
                print(f"  Contains NaN/Inf: {coords_nan}")
                print(f"  RMSD: {rmsd:.3f} Å")

            # Compute strain if SMILES is available
            strain = None
            reason = None
            if smiles_list is not None and g < len(smiles_list):
                mol = Chem.MolFromSmiles(smiles_list[g])
                if mol is not None:
                    # Sanity check against preprocessing metadata if available
                    metadata_match = True
                    if hasattr(batch, "ligand_meta_atom_count"):
                        meta_data = batch.ligand_meta_atom_count
                        if isinstance(meta_data, torch.Tensor):
                            meta_count = meta_data[g].item()
                        elif isinstance(meta_data, list):
                            meta_count = meta_data[g]
                        else:
                            meta_count = meta_data

                        if mol.GetNumAtoms() != meta_count:
                            metadata_match = False
                    
                    if metadata_match:
                        strain, reason = strain_energy_ratio(mol, gen_g.numpy())
                        if reason is not None:
                            if "invalid_pose_geometry" in reason:
                                strain_failures["invalid_pose_geometry"] += 1
                            elif "rdkit_mmff_setup_failure" in reason:
                                strain_failures["rdkit_mmff_setup_failure"] += 1
                            elif "absurd_energy" in reason:
                                strain_failures["absurd_energy"] += 1
                            else:
                                strain_failures["other"] += 1
            strain_all.append(strain)

            if dump_eval_predictions:
                eval_dumps.append({
                    "complex_id": complex_id,
                    "rmsd": float(rmsd) if not np.isnan(rmsd) else None,
                    "strain": float(strain) if strain is not None else None,
                    "chemistry_failure": reason if reason is not None else "success",
                    "pred_coords_min": float(gen_g.min().item()),
                    "pred_coords_max": float(gen_g.max().item()),
                })
            
            debug_count += 1

    rmsd_arr = np.array(rmsd_all)
    valid_strains = [s for s in strain_all if s is not None]

    n_total_strain_attempted = sum(strain_failures.values()) + len(valid_strains)

    if dump_eval_predictions and eval_dumps:
        out_path = dump_eval_predictions
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            for item in eval_dumps:
                f.write(json.dumps(item) + "\n")
        print(f"\nDumped {len(eval_dumps)} predictions to {out_path}")

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
        "n_mmff_failed":     len(strain_all) - len(valid_strains) - strain_failures.get("absurd_energy", 0),
        "n_total":           len(rmsd_all),
        "strain_failures":   strain_failures
    }


def compute_etkdg_baseline(test_loader, mol_lookup: Dict) -> Dict:
    """
    Compute ETKDG baseline metrics.
    """
    rmsd_all = []
    strain_all = []
    failures = {
        "n_success": 0,
        "n_atom_mismatch": 0,
        "n_embed_fail": 0,
        "n_other_fail": 0
    }
    
    strain_failures = {
        "invalid_pose_geometry": 0,
        "rdkit_mmff_setup_failure": 0,
        "absurd_energy": 0,
        "other": 0
    }

    for batch in test_loader:
        # batch.complex_id is a list of strings in a batched HeteroData
        ids = batch.complex_id if isinstance(batch.complex_id, list) else [batch.complex_id]
        crystal_pos_list = _unbatch_positions(
            batch["ligand"].pos, batch["ligand"].batch
        )

        for cid, crystal_pos in zip(ids, crystal_pos_list):
            mol = mol_lookup.get(cid)
            if mol is None:
                failures["n_other_fail"] += 1
                continue

            if mol.GetNumAtoms() != crystal_pos.shape[0]:
                failures["n_atom_mismatch"] += 1
                continue

            try:
                mol_h = Chem.AddHs(mol)
                ret = AllChem.EmbedMolecule(mol_h, randomSeed=42)
                if ret != 0:
                    failures["n_embed_fail"] += 1
                    continue

                AllChem.MMFFOptimizeMolecule(mol_h)
                mol_h = Chem.RemoveHs(mol_h)
                etkdg_pos = torch.tensor(
                    mol_h.GetConformer().GetPositions(), dtype=torch.float32
                )
            except Exception:
                failures["n_embed_fail"] += 1
                continue

            rmsd = kabsch_rmsd(etkdg_pos, crystal_pos.cpu())
            if np.isnan(rmsd):
                failures["n_atom_mismatch"] += 1
                continue
                
            rmsd_all.append(rmsd)
            failures["n_success"] += 1

            ratio, reason = strain_energy_ratio(mol, etkdg_pos.numpy())
            if reason is not None:
                if "invalid_pose_geometry" in reason:
                    strain_failures["invalid_pose_geometry"] += 1
                elif "rdkit_mmff_setup_failure" in reason:
                    strain_failures["rdkit_mmff_setup_failure"] += 1
                elif "absurd_energy" in reason:
                    strain_failures["absurd_energy"] += 1
                else:
                    strain_failures["other"] += 1
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
        "n_mmff_failed":     len(strain_all) - len(valid_strains) - strain_failures.get("absurd_energy", 0),
        "n_total":           len(rmsd_all),
        "failures":          failures,
        "strain_failures":   strain_failures
    }


def _unbatch_positions(pos: Tensor, batch: Tensor) -> List[Tensor]:
    """Split a batched position tensor into per-graph tensors."""
    n_graphs = batch.max().item() + 1
    return [pos[batch == g] for g in range(n_graphs)]
