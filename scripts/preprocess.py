"""
Preprocess raw PDBBind entries (PDB protein + SDF ligand) into
PyTorch Geometric HeteroData .pt files.

Usage:
    python scripts/preprocess.py \
        --raw_dir    data/raw \
        --out_dir    data/processed \
        --pocket_cutoff 10.0 \
        --cross_k    4 \
        --n_rbf      16 \
        --n_workers  4

Outputs:
    data/processed/<pdb_id>.pt   — one HeteroData per complex
    data/splits.json             — train/val/test split (seed=42)
    data/filter_log.csv          — per-complex filter outcome
"""

import argparse
import json
import os
import random
import sys
import traceback
from multiprocessing import Pool, cpu_count
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem
import MDAnalysis as mda

from src.data.featurize import (
    build_cross_edges,
    featurize_ligand,
    featurize_pocket,
    passes_filters,
)


# ---------------------------------------------------------------------------
# Processing one complex
# ---------------------------------------------------------------------------

def process_one(args_tuple) -> dict:
    """
    Process a single PDBBind complex.

    Args:
        args_tuple: (pdb_id, raw_dir, out_dir, pocket_cutoff, cross_k, n_rbf)

    Returns:
        status dict for logging
    """
    pdb_id, raw_dir, out_dir, pocket_cutoff, cross_k, n_rbf = args_tuple

    status = {"id": pdb_id, "status": "ok", "reason": "", "n_lig": 0, "n_poc": 0, "mw": 0.0}

    try:
        complex_dir = os.path.join(raw_dir, pdb_id)

        # Locate protein PDB
        protein_pdb = _find_file(complex_dir, f"{pdb_id}_protein.pdb")
        if protein_pdb is None:
            protein_pdb = _find_file(complex_dir, "_protein.pdb")
        if protein_pdb is None:
            status["status"] = "skip"
            status["reason"] = "no_protein_pdb"
            return status

        # Locate ligand SDF
        ligand_sdf = _find_file(complex_dir, f"{pdb_id}_ligand.sdf")
        if ligand_sdf is None:
            ligand_sdf = _find_file(complex_dir, "_ligand.sdf")
        if ligand_sdf is None:
            status["status"] = "skip"
            status["reason"] = "no_ligand_sdf"
            return status

        # Load and sanitize ligand
        mol = Chem.MolFromMolFile(ligand_sdf, removeHs=True, sanitize=True)
        if mol is None:
            status["status"] = "skip"
            status["reason"] = "rdkit_parse_failed"
            return status

        # Apply filters
        passed, filter_details = passes_filters(mol)
        status["mw"] = filter_details.get("mw", 0.0)
        if not passed:
            reasons = [k for k in ("mw_ok", "rot_bonds_ok", "heavy_atoms_ok", "no_metal_ok")
                       if not filter_details.get(k, True)]
            status["status"] = "filtered"
            status["reason"] = ",".join(reasons)
            return status

        # Load protein with MDAnalysis
        u = mda.Universe(protein_pdb)

        # Extract pocket atoms within pocket_cutoff of ligand centroid
        lig_conf = mol.GetConformer()
        lig_coords = lig_conf.GetPositions()
        centroid = lig_coords.mean(axis=0)

        pocket_selection = (
            f"protein and (point {centroid[0]:.3f} {centroid[1]:.3f} {centroid[2]:.3f} {pocket_cutoff})"
        )
        pocket_atoms = u.select_atoms(pocket_selection)

        if len(pocket_atoms) < 10:
            status["status"] = "skip"
            status["reason"] = "pocket_too_small"
            return status

        # Featurize
        lig_h, lig_ei, lig_ea, lig_pos = featurize_ligand(mol)
        poc_h, poc_ei, poc_ea, poc_pos = featurize_pocket(pocket_atoms, n_rbf=n_rbf)
        cross_ei, cross_ea = build_cross_edges(lig_pos, poc_pos, k=cross_k, n_rbf=n_rbf)

        # Assemble HeteroData
        data = HeteroData()
        data["ligand"].x = lig_h
        data["ligand"].pos = lig_pos
        data["ligand"].edge_index = lig_ei
        data["ligand"].edge_attr = lig_ea

        data["pocket"].x = poc_h
        data["pocket"].pos = poc_pos
        data["pocket"].edge_index = poc_ei
        data["pocket"].edge_attr = poc_ea

        data["pocket", "to", "ligand"].edge_index = cross_ei
        data["pocket", "to", "ligand"].edge_attr = cross_ea

        data.complex_id = pdb_id
        data.smiles = Chem.MolToSmiles(mol)
        data.ligand_meta_canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        data.ligand_meta_atom_count = mol.GetNumAtoms()
        data.ligand_meta_heavy_atom_count = mol.GetNumHeavyAtoms()
        data.ligand_meta_ordered_symbols = ",".join([a.GetSymbol() for a in mol.GetAtoms()])

        # Save
        out_path = os.path.join(out_dir, f"{pdb_id}.pt")
        torch.save(data, out_path)

        status["n_lig"] = lig_h.shape[0]
        status["n_poc"] = poc_h.shape[0]

    except Exception:
        status["status"] = "error"
        status["reason"] = traceback.format_exc()[:200]

    return status


def _find_file(directory: str, suffix: str) -> Optional[str]:
    """Find a file in directory whose name ends with suffix."""
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if fname.endswith(suffix):
            return os.path.join(directory, fname)
    return None


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def load_release_years(index_file: str) -> dict:
    """Parse PDBBind index file (e.g. INDEX_refined_data.2020) → {pdb_id: year}."""
    years = {}
    with open(index_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                pdb_id = parts[0].lower()
                try:
                    years[pdb_id] = int(parts[2])
                except ValueError:
                    pass
    return years


def make_splits(
    ok_ids: list,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    release_years: dict = None,
    test_year: int = 2019,
) -> dict:
    """
    Time-split if release_years provided (matches DiffDock benchmark):
      test  = complexes with release year >= test_year
      val   = random 10% of remaining train
      train = rest
    Falls back to random split if release_years is None.
    """
    if release_years is not None:
        test_ids = [i for i in ok_ids if release_years.get(i, 0) >= test_year]
        remaining = [i for i in ok_ids if i not in set(test_ids)]
        rng = random.Random(seed)
        rng.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_frac))
        val_ids = remaining[:n_val]
        train_ids = remaining[n_val:]
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    rng = random.Random(seed)
    shuffled = list(ok_ids)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, int(n_total * test_frac))
    n_val = max(1, int(n_total * val_frac))
    test_ids = shuffled[:n_test]
    val_ids = shuffled[n_test:n_test + n_val]
    train_ids = shuffled[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBBind to .pt HeteroData")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--pocket_cutoff", type=float, default=10.0)
    parser.add_argument("--cross_k", type=int, default=4)
    parser.add_argument("--n_rbf", type=int, default=16)
    parser.add_argument("--n_workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--splits_out", default="data/splits.json")
    parser.add_argument("--log_out", default="data/filter_log.csv")
    parser.add_argument("--index_file", default=None,
                        help="PDBBind index file for time-split (e.g. INDEX_refined_data.2020)")
    parser.add_argument("--test_year", type=int, default=2019,
                        help="Complexes with release year >= this go to test (default: 2019)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover all complex directories
    pdb_ids = sorted([
        d for d in os.listdir(args.raw_dir)
        if os.path.isdir(os.path.join(args.raw_dir, d))
    ])

    if not pdb_ids:
        print(f"No complex directories found in {args.raw_dir}")
        print("Download PDBBind data first: python scripts/download_pdbbind_core.py")
        return

    print(f"Found {len(pdb_ids)} complexes in {args.raw_dir}")

    task_args = [
        (pid, args.raw_dir, args.out_dir, args.pocket_cutoff, args.cross_k, args.n_rbf)
        for pid in pdb_ids
    ]

    # Process (parallel on non-Windows, sequential on Windows with n_workers=1)
    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            results = list(pool.imap_unordered(process_one, task_args, chunksize=10))
    else:
        results = [process_one(t) for t in task_args]

    # Save filter log
    df = pd.DataFrame(results)
    df.to_csv(args.log_out, index=False)

    ok_ids = [r["id"] for r in results if r["status"] == "ok"]
    print(f"\nResults: {len(ok_ids)} ok / {len(results)} total")
    print(df["status"].value_counts().to_string())

    if ok_ids:
        release_years = None
        if args.index_file:
            release_years = load_release_years(args.index_file)
            print(f"Time-split: using release years from {args.index_file}, test_year >= {args.test_year}")
        splits = make_splits(ok_ids, release_years=release_years, test_year=args.test_year)
        with open(args.splits_out, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplit: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")
        print(f"Saved splits → {args.splits_out}")
        print(f"Saved log    → {args.log_out}")
    else:
        print("No complexes passed. Check filter_log.csv for details.")


if __name__ == "__main__":
    # Windows multiprocessing requires spawn
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
