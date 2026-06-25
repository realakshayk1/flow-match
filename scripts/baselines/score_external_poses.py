"""
Shared evaluator for EXTERNAL docking baselines (DiffDock, Vina, GNINA, ...).

Given a directory of predicted ligand poses (SDF) and matching reference crystal ligands (SDF),
compute the SAME in-frame, symmetry-corrected dock RMSD used for the Flow-Match model and,
optionally, PoseBusters validity — so every method lands in one comparable table.

Unlike the model path (where atom order is canonicalized and identical between pred and crystal),
external tools emit their own atom ordering, so cross-molecule RMSD is computed with RDKit
``rdMolAlign.CalcRMS`` (no superposition, minimized over atom mappings).

Inputs:
  --pred_dir : directory of predicted SDFs, one per complex. File stem (optionally with a
               --pred_suffix like "_rank1") is the complex_id.
  --ref_dir  : directory of reference crystal ligand SDFs named "{complex_id}.sdf".
  --method   : label for the output (e.g. "diffdock", "vina", "gnina").

Output: {out_dir}/{method}_summary.json with the same headline keys as eval_posebusters
(dock_rmsd_median, dock_rmsd_pct_under_2A, pb_valid_pct, rmsd_lt2_and_pbvalid_pct).

Example:
  python scripts/baselines/score_external_poses.py --method diffdock \
      --pred_dir diffdock_out/sdf --pred_suffix _rank1 \
      --ref_dir eval/crystal_ligands --out_dir results/phase3/baselines
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_first_mol(path):
    supplier = Chem.SDMolSupplier(path, sanitize=True, removeHs=True)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def inframe_rms(pred_mol, ref_mol):
    """In-frame, symmetry-corrected RMS between two conformers of the same molecule.

    Uses rdMolAlign.CalcRMS (no alignment; minimizes over atom mappings). Returns NaN if the
    molecules are not comparable (different graphs).
    """
    try:
        # CalcRMS(prbMol, refMol) maps probe onto ref by substructure; no rigid-body alignment.
        return float(rdMolAlign.CalcRMS(pred_mol, ref_mol))
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, help="Baseline label, e.g. diffdock / vina / gnina")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--pred_suffix", default="", help="Suffix on predicted filenames, e.g. _rank1")
    p.add_argument("--success_rmsd", type=float, default=2.0)
    p.add_argument("--run_posebusters", action="store_true",
                   help="Also compute PB-valid (requires the posebusters package).")
    p.add_argument("--out_dir", default="results/phase3/baselines")
    args = p.parse_args()

    pb = None
    if args.run_posebusters:
        from posebusters import PoseBusters
        pb = PoseBusters(config="mol")

    rows = []
    for pred_path in sorted(glob.glob(os.path.join(args.pred_dir, "*.sdf"))):
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        cid = stem[: -len(args.pred_suffix)] if args.pred_suffix and stem.endswith(args.pred_suffix) else stem
        ref_path = os.path.join(args.ref_dir, f"{cid}.sdf")
        if not os.path.exists(ref_path):
            rows.append({"complex_id": cid, "dock_rmsd": float("nan"), "error": "no_reference"})
            continue
        pred_mol = _load_first_mol(pred_path)
        ref_mol = _load_first_mol(ref_path)
        if pred_mol is None or ref_mol is None:
            rows.append({"complex_id": cid, "dock_rmsd": float("nan"), "error": "load_failed"})
            continue

        dock_rmsd = inframe_rms(pred_mol, ref_mol)
        row = {"complex_id": cid, "dock_rmsd": dock_rmsd}
        if pb is not None:
            try:
                res = pb.bust(mol_pred=pred_mol).iloc[0].to_dict()
                checks = [v for k, v in res.items() if isinstance(v, (bool, np.bool_))]
                row["pb_valid"] = bool(all(checks)) if checks else None
            except Exception as e:
                row["pb_valid"] = None
                row["pb_error"] = str(e)
        rows.append(row)

    dock = np.array([r["dock_rmsd"] for r in rows if not np.isnan(r["dock_rmsd"])])
    summary = {
        "method": args.method,
        "n_total": len(rows),
        "n_scored": int(len(dock)),
        "dock_rmsd_median": round(float(np.median(dock)), 3) if len(dock) else None,
        "dock_rmsd_pct_under_2A": round(float((dock < args.success_rmsd).mean() * 100), 1) if len(dock) else None,
    }
    if args.run_posebusters:
        pv = [r.get("pb_valid") for r in rows if r.get("pb_valid") is not None]
        summary["pb_valid_pct"] = round(float(np.mean(pv) * 100), 1) if pv else None
        joint = [(not np.isnan(r["dock_rmsd"]) and r["dock_rmsd"] < args.success_rmsd and r.get("pb_valid"))
                 for r in rows if r.get("pb_valid") is not None]
        summary["rmsd_lt2_and_pbvalid_pct"] = round(float(np.mean(joint) * 100), 1) if joint else None

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.method}_summary.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_complex": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
