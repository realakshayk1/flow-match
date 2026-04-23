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
from src.models.egnn import build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import kabsch_rmsd, mol_with_coords, uff_minimize


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


def run_posebusters_on_sdf(pb: PoseBusters, sdf_path: str) -> dict:
    """Run PoseBusters on a single SDF file. Returns dict of check results."""
    try:
        results = pb.bust(sdf_path)
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
                        help="Run UFF minimization on generated poses before PB checks")
    parser.add_argument("--n_inference_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    args = parser.parse_args()

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
    model.load_state_dict(ckpt["model_state"])
    flow_matcher = FlowMatcher(model, n_steps=args.n_inference_steps).to(device)
    flow_matcher.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, n_steps={args.n_inference_steps}")

    # Load data
    splits = load_splits(args.splits)
    dataset = PDBBindDataset(args.processed_dir, splits[args.split])
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Running on {len(dataset)} {args.split} complexes")

    # PoseBusters in redock mode (ligand-only, no protein PDB needed)
    pb = PoseBusters(config="redock")

    all_rows = []
    rmsd_all = []

    for batch in tqdm(loader, desc="Generating poses"):
        batch = batch.to(device)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = int(lig_batch.max().item()) + 1

        generated = flow_matcher.generate(batch, n_steps=args.n_inference_steps)

        smiles_list = getattr(batch, "ligand_meta_canonical_smiles", None)
        complex_ids = batch.complex_id if isinstance(batch.complex_id, list) else [batch.complex_id]

        for g in range(n_graphs):
            mask = lig_batch == g
            crystal_g = crystal_pos[mask].cpu().numpy()
            gen_g = generated[g].cpu().numpy()
            cid = complex_ids[g]
            smiles = smiles_list[g] if smiles_list is not None else None

            # RMSD against crystal
            rmsd = kabsch_rmsd(
                torch.tensor(gen_g), torch.tensor(crystal_g)
            )
            rmsd_all.append(rmsd)

            if smiles is None:
                all_rows.append({"complex_id": cid, "rmsd": rmsd, "pb_error": "no_smiles"})
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                all_rows.append({"complex_id": cid, "rmsd": rmsd, "pb_error": "invalid_smiles"})
                continue

            coords_to_save = gen_g
            uff_failed = False

            if args.uff_postprocess:
                opt_coords, uff_reason = uff_minimize(mol, gen_g)
                if opt_coords is not None:
                    coords_to_save = opt_coords
                else:
                    uff_failed = True

            # Write SDF
            sdf_path = os.path.join(poses_dir, f"{cid}_pred.sdf")
            write_ok = mol_to_sdf(mol, coords_to_save, sdf_path)

            if not write_ok:
                all_rows.append({"complex_id": cid, "rmsd": rmsd, "pb_error": "sdf_write_failed"})
                continue

            # Run PoseBusters
            pb_row = run_posebusters_on_sdf(pb, sdf_path)
            pb_row["complex_id"] = cid
            pb_row["rmsd"] = rmsd
            pb_row["uff_postprocess"] = args.uff_postprocess
            pb_row["uff_failed"] = uff_failed
            all_rows.append(pb_row)

    # Aggregate
    df = pd.DataFrame(all_rows)
    raw_csv = os.path.join(args.output_dir, "results_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\nSaved per-complex results to {raw_csv}")

    # PB-valid = passes all checks (True in every bool column except errors)
    check_cols = [c for c in df.columns
                  if c not in ("complex_id", "rmsd", "uff_postprocess", "uff_failed", "pb_error")
                  and df[c].dtype == bool]

    if check_cols:
        df["pb_valid"] = df[check_cols].all(axis=1)
        pb_valid_rate = df["pb_valid"].mean() * 100

        summary = {
            "n_total": len(df),
            "n_pb_valid": int(df["pb_valid"].sum()),
            "pb_valid_pct": round(pb_valid_rate, 1),
            "rmsd_median": round(float(np.median(rmsd_all)), 3),
            "rmsd_pct_under_2A": round(float((np.array(rmsd_all) < 2.0).mean() * 100), 1),
            "uff_postprocess": args.uff_postprocess,
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

    print(f"\n{'='*50}")
    print(f"PoseBusters Results ({'with UFF' if args.uff_postprocess else 'raw model output'})")
    print(f"{'='*50}")
    if "pb_valid_pct" in summary:
        print(f"PB-valid:      {summary['pb_valid_pct']:.1f}%  ({summary['n_pb_valid']}/{summary['n_total']})")
        print(f"RMSD median:   {summary['rmsd_median']:.3f} Å")
        print(f"RMSD < 2 Å:    {summary['rmsd_pct_under_2A']:.1f}%")
        print(f"\nPer-check pass rates:")
        for check, rate in sorted(summary["per_check_pass_rate"].items()):
            flag = " ← failing" if rate < 90 else ""
            print(f"  {check:<50} {rate:.1f}%{flag}")
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
