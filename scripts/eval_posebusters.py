"""
PoseBusters evaluation for Flow-Match model.

Usage:
    python scripts/eval_posebusters.py \
        --checkpoint checkpoints/best.pt \
        --processed_dir data/processed \
        --splits data/splits.json \
        --output_dir eval/posebusters \
        [--uff_postprocess] \
        [--n_inference_steps 100] \
        [--device cpu]

Outputs:
    eval/posebusters/results_raw.csv       — per-complex per-check results
    eval/posebusters/results_summary.json  — aggregated PB-valid rates
    eval/posebusters/poses/                — SDF files of generated poses
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from posebusters import PoseBusters
from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.models.egnn import build_default_model, load_model_state
from src.models.flow_model import FlowMatcher
from src.training.metrics import (
    kabsch_rmsd, symmetry_rmsd, mol_with_coords, uff_minimize, pocket_aware_relax,
)


def mol_to_sdf(mol: Chem.Mol, coords: np.ndarray, path: str) -> bool:
    """Write mol with given coords to SDF. Returns True on success."""
    try:
        m = mol_with_coords(mol, coords)
        writer = Chem.SDWriter(path)
        writer.write(m)
        writer.close()
        return True
    except Exception:
        return False


def run_posebusters(pb: PoseBusters, mol_pred: Chem.Mol) -> dict:
    """Run PoseBusters on one predicted molecule. Returns dict of check results."""
    try:
        results = pb.bust(mol_pred=mol_pred)
        # results is a DataFrame with one row per molecule
        row = results.iloc[0].to_dict()
        return row
    except Exception as e:
        return {"pb_error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output_dir", default="eval/posebusters")
    parser.add_argument("--uff_postprocess", action="store_true",
                        help="[deprecated] alias for --relax uff")
    parser.add_argument("--relax", choices=["none", "uff", "pocket"], default="pocket",
                        help="Post-generation relaxation: 'none' raw, 'uff' plain UFF, "
                             "'pocket' restrained MMFF toward the predicted pose (default).")
    parser.add_argument("--n_inference_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Optional cap on number of complexes for quick validation runs")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument(
        "--export_phase1_dir",
        default=None,
        help=(
            "Optional directory for interview-style bundle: copies results_summary.json "
            "to summary.json and results_raw.csv to per_complex.csv (PDBBind = per-complex)."
        ),
    )
    args = parser.parse_args()

    # Back-compat: --uff_postprocess forces the UFF relaxation mode.
    if args.uff_postprocess:
        args.relax = "uff"
    relax_enabled = args.relax != "none"

    os.makedirs(args.output_dir, exist_ok=True)
    poses_dir = os.path.join(args.output_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)

    # Load model
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = ckpt.get("run_config", {})
    
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = run_config.get("hidden_dim", ckpt["model_state"]["lig_emb.weight"].shape[0] if "lig_emb.weight" in ckpt["model_state"] else 128)
        
    n_layers = args.n_layers
    if n_layers is None:
        layer_indices = [int(k.split(".")[1]) for k in ckpt["model_state"].keys() if k.startswith("layers.")]
        n_layers = run_config.get("n_layers", max(layer_indices) + 1 if layer_indices else 6)

    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    load_model_state(model, ckpt["model_state"])
    flow_matcher = FlowMatcher(model, n_steps=args.n_inference_steps).to(device)
    flow_matcher.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, n_steps={args.n_inference_steps}")

    # Load data
    splits = load_splits(args.splits)
    dataset = PDBBindDataset(args.processed_dir, splits[args.split])
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    n_target = min(len(dataset), args.max_complexes) if args.max_complexes else len(dataset)
    print(f"Running on {n_target} {args.split} complexes")

    # PoseBusters molecular validity mode (ligand-only checks).
    pb = PoseBusters(config="mol")

    all_rows = []
    rmsd_all = []       # shape RMSD (Kabsch) — kept for continuity
    dock_rmsd_all = []  # in-frame symmetry-corrected RMSD — the docking metric

    processed = 0
    stop_early = False
    for batch in tqdm(loader, desc="Generating poses"):
        if stop_early:
            break
        batch = batch.to(device)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = int(lig_batch.max().item()) + 1

        generated = flow_matcher.generate(batch, n_steps=args.n_inference_steps)

        smiles_list = getattr(batch, "ligand_meta_canonical_smiles", None)
        complex_ids = batch.complex_id if isinstance(batch.complex_id, list) else [batch.complex_id]

        for g in range(n_graphs):
            if args.max_complexes is not None and processed >= args.max_complexes:
                stop_early = True
                break
            mask = lig_batch == g
            crystal_g = crystal_pos[mask].cpu().numpy()
            gen_g = generated[g].cpu().numpy()
            cid = complex_ids[g]
            smiles = smiles_list[g] if smiles_list is not None else None

            # Shape RMSD (Kabsch-aligned) — conformer fidelity, not placement.
            shape_rmsd = kabsch_rmsd(
                torch.tensor(gen_g), torch.tensor(crystal_g)
            )
            rmsd_all.append(shape_rmsd)

            if smiles is None:
                dock_rmsd_all.append(float("nan"))
                all_rows.append({"complex_id": cid, "rmsd": float("nan"),
                                 "shape_rmsd": shape_rmsd, "dock_rmsd": float("nan"),
                                 "pb_error": "no_smiles"})
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                dock_rmsd_all.append(float("nan"))
                all_rows.append({"complex_id": cid, "rmsd": float("nan"),
                                 "shape_rmsd": shape_rmsd, "dock_rmsd": float("nan"),
                                 "pb_error": "invalid_smiles"})
                continue

            # In-frame, symmetry-corrected docking RMSD (no superposition). This is the
            # headline "rmsd" used downstream — comparable to DiffDock/Vina.
            if mol.GetNumAtoms() == gen_g.shape[0]:
                dock_rmsd = symmetry_rmsd(mol, gen_g, crystal_g)
            else:
                dock_rmsd = float("nan")
            dock_rmsd_all.append(dock_rmsd)
            rmsd = dock_rmsd

            coords_to_save = gen_g
            relax_failed = False

            if args.relax == "uff":
                opt_coords, _ = uff_minimize(mol, gen_g)
                if opt_coords is not None:
                    coords_to_save = opt_coords
                else:
                    relax_failed = True
            elif args.relax == "pocket":
                opt_coords, _ = pocket_aware_relax(mol, gen_g)
                if opt_coords is not None:
                    coords_to_save = opt_coords
                else:
                    relax_failed = True

            # Recompute dock RMSD on the relaxed coordinates actually scored by PoseBusters,
            # so the reported RMSD matches the pose that was validated.
            if relax_enabled and not relax_failed and mol.GetNumAtoms() == coords_to_save.shape[0]:
                dock_rmsd = symmetry_rmsd(mol, np.asarray(coords_to_save), crystal_g)
                dock_rmsd_all[-1] = dock_rmsd
                rmsd = dock_rmsd

            # Write SDF
            sdf_path = os.path.join(poses_dir, f"{cid}_pred.sdf")
            write_ok = mol_to_sdf(mol, coords_to_save, sdf_path)

            if not write_ok:
                all_rows.append({"complex_id": cid, "rmsd": rmsd, "pb_error": "sdf_write_failed"})
                continue

            mol_pred = mol_with_coords(mol, coords_to_save)
            pb_row = run_posebusters(pb, mol_pred=mol_pred)
            pb_row["complex_id"] = cid
            pb_row["rmsd"] = rmsd
            pb_row["shape_rmsd"] = shape_rmsd
            pb_row["dock_rmsd"] = dock_rmsd
            pb_row["relax"] = args.relax
            pb_row["uff_postprocess"] = relax_enabled  # back-compat column
            pb_row["uff_failed"] = relax_failed
            all_rows.append(pb_row)
            processed += 1

    # Aggregate
    df = pd.DataFrame(all_rows)
    raw_csv = os.path.join(args.output_dir, "results_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\nSaved per-complex results to {raw_csv}")

    # PB-valid = passes all checks (True in every bool column except errors)
    meta_cols = ("complex_id", "rmsd", "shape_rmsd", "dock_rmsd",
                 "relax", "uff_postprocess", "uff_failed", "pb_error")
    check_cols = [c for c in df.columns
                  if c not in meta_cols and df[c].dtype == bool]

    dock_arr = np.array([r for r in dock_rmsd_all if not np.isnan(r)])
    shape_arr = np.array([r for r in rmsd_all if not np.isnan(r)])

    if check_cols:
        df["pb_valid"] = df[check_cols].all(axis=1)
        pb_valid_rate = df["pb_valid"].mean() * 100

        # Headline joint metric: fraction with dock RMSD < 2A AND physically valid.
        if "dock_rmsd" in df.columns:
            joint_mask = (df["dock_rmsd"] < 2.0) & df["pb_valid"]
            joint_pct = round(float(joint_mask.mean() * 100), 1)
        else:
            joint_pct = None

        summary = {
            "n_total": len(df),
            "n_pb_valid": int(df["pb_valid"].sum()),
            "pb_valid_pct": round(pb_valid_rate, 1),
            # Headline docking metric: in-frame, symmetry-corrected RMSD.
            "dock_rmsd_median": round(float(np.median(dock_arr)), 3) if len(dock_arr) else None,
            "dock_rmsd_pct_under_2A": round(float((dock_arr < 2.0).mean() * 100), 1) if len(dock_arr) else None,
            "n_dock_rmsd_valid": int(len(dock_arr)),
            # The single number to report: joint pose accuracy + physical validity.
            "rmsd_lt2_and_pbvalid_pct": joint_pct,
            # Shape RMSD (Kabsch) kept for reference / continuity with older runs.
            "shape_rmsd_median": round(float(np.median(shape_arr)), 3) if len(shape_arr) else None,
            "shape_rmsd_pct_under_2A": round(float((shape_arr < 2.0).mean() * 100), 1) if len(shape_arr) else None,
            # Legacy keys (now point at the in-frame docking RMSD).
            "rmsd_median": round(float(np.median(dock_arr)), 3) if len(dock_arr) else None,
            "rmsd_pct_under_2A": round(float((dock_arr < 2.0).mean() * 100), 1) if len(dock_arr) else None,
            "relax": args.relax,
            "uff_postprocess": relax_enabled,
            "n_inference_steps": args.n_inference_steps,
            "per_check_pass_rate": {
                col: round(df[col].mean() * 100, 1)
                for col in sorted(check_cols)
            }
        }
    else:
        summary = {"error": "no PB check columns found", "n_total": len(df)}

    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.export_phase1_dir:
        bundle = Path(args.export_phase1_dir).resolve()
        bundle.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, bundle / "summary.json")
        shutil.copy2(raw_csv, bundle / "per_complex.csv")
        print(f"Phase 1 bundle: {bundle / 'summary.json'}, {bundle / 'per_complex.csv'}")

    print(f"\n{'='*50}")
    print(f"PoseBusters Results (relax={args.relax})")
    print(f"{'='*50}")
    if "pb_valid_pct" in summary:
        print(f"PB-valid:               {summary['pb_valid_pct']:.1f}%  ({summary['n_pb_valid']}/{summary['n_total']})")
        if summary.get("dock_rmsd_median") is not None:
            print(f"Dock RMSD median:       {summary['dock_rmsd_median']:.3f} A  (in-frame, symmetry-corrected)")
            print(f"Dock RMSD < 2 A:        {summary['dock_rmsd_pct_under_2A']:.1f}%")
        if summary.get("rmsd_lt2_and_pbvalid_pct") is not None:
            print(f"RMSD<2A AND PB-valid:   {summary['rmsd_lt2_and_pbvalid_pct']:.1f}%   <-- headline")
        if summary.get("shape_rmsd_median") is not None:
            print(f"Shape RMSD median:      {summary['shape_rmsd_median']:.3f} A  (Kabsch-aligned; not docking)")
        print(f"\nPer-check pass rates:")
        for check, rate in sorted(summary["per_check_pass_rate"].items()):
            flag = " [failing]" if rate < 90 else ""
            print(f"  {check:<50} {rate:.1f}%{flag}")
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
