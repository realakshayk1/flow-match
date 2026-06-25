import argparse
import csv
import math
import os
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.crossdock.common import ensure_dir, load_first_mol, mol_coords, write_json, write_jsonl, write_pose_sdf
from src.data.featurize import build_cross_edges, featurize_ligand, featurize_pocket
from src.models.egnn import build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import kabsch_rmsd


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_pocket(universe: mda.Universe, center: np.ndarray, radius: float, lig_pos: torch.Tensor):
    cx, cy, cz = center.tolist()
    sel = f"protein and (point {cx:.3f} {cy:.3f} {cz:.3f} {radius:.3f})"
    atoms = universe.select_atoms(sel)
    if len(atoms) < 10:
        return None
    poc_h, poc_edge_index, poc_edge_attr, poc_pos = featurize_pocket(atoms, dist_cutoff=6.0, n_rbf=16)
    cross_edge_index, cross_edge_attr = build_cross_edges(lig_pos, poc_pos, k=4, n_rbf=16)
    return poc_h, poc_edge_index, poc_edge_attr, poc_pos, cross_edge_index, cross_edge_attr


def _run_one(
    flow_matcher: FlowMatcher,
    row: dict[str, str],
    out_dir: Path,
    device: torch.device,
    n_steps: int,
    pocket_radius: float,
    dry_run: bool,
) -> dict[str, object]:
    pair_id = row["pair_id"]
    target_id = row["target_id"]
    receptor_pdb = Path(row["receptor_pdb"])
    ref_sdf = Path(row["reference_ligand_sdf"])
    lig_sdf = Path(row["ligand_input_sdf"])
    native_sdf = Path(row["receptor_native_ligand_sdf"])

    result: dict[str, object] = {"pair_id": pair_id, "target_id": target_id, "engine": "model"}
    try:
        lig_mol = load_first_mol(lig_sdf)
        ref_mol = load_first_mol(ref_sdf)
        ref_coords = torch.tensor(mol_coords(ref_mol), dtype=torch.float32)

        out_pose = out_dir / "poses" / f"{pair_id}_model_pred.sdf"
        if dry_run:
            coords = mol_coords(ref_mol) + np.random.default_rng(13).normal(0.0, 0.15, size=mol_coords(ref_mol).shape)
            write_pose_sdf(lig_mol, coords.astype(np.float32), out_pose)
            pred_mol = load_first_mol(out_pose)
            rmsd = float(kabsch_rmsd(torch.tensor(mol_coords(pred_mol), dtype=torch.float32), ref_coords))
            result.update({"status": "ok", "pred_pose_sdf": str(out_pose), "rmsd": rmsd})
            return result

        native_mol = load_first_mol(native_sdf)
        native_center = np.mean(mol_coords(native_mol), axis=0)
        lig_h, lig_edge_index, lig_edge_attr, lig_pos = featurize_ligand(lig_mol)

        universe = mda.Universe(str(receptor_pdb))
        pocket_graph = _extract_pocket(universe, native_center, pocket_radius, lig_pos)
        if pocket_graph is None:
            result.update({"status": "failed", "error": "too_few_pocket_atoms", "rmsd": None})
            return result

        poc_h, poc_edge_index, poc_edge_attr, poc_pos, cross_edge_index, cross_edge_attr = pocket_graph
        poc_center = poc_pos.mean(dim=0)
        poc_pos_centered = poc_pos - poc_center
        with torch.no_grad():
            pred = flow_matcher.generate_single(
                lig_h=lig_h.to(device),
                poc_x=poc_pos_centered.to(device),
                poc_h=poc_h.to(device),
                lig_edge_index=lig_edge_index.to(device),
                lig_edge_attr=lig_edge_attr.to(device),
                poc_edge_index=poc_edge_index.to(device),
                poc_edge_attr=poc_edge_attr.to(device),
                cross_edge_index=cross_edge_index.to(device),
                cross_edge_attr=cross_edge_attr.to(device),
                n_steps=n_steps,
            )
        pred_world = (pred + poc_center.to(device)).cpu().numpy()
        write_pose_sdf(lig_mol, pred_world, out_pose)
        pred_mol = load_first_mol(out_pose)
        pred_coords = torch.tensor(mol_coords(pred_mol), dtype=torch.float32)
        rmsd = float(kabsch_rmsd(pred_coords, ref_coords))
        result.update({"status": "ok", "pred_pose_sdf": str(out_pose), "rmsd": rmsd})
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc), "rmsd": None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow-Match CASF cross-docking")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--checkpoint", default="", help="Required unless --dry_run")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--pocket_radius", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_pairs", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = _read_manifest(Path(args.manifest_csv).resolve())
    if args.limit_pairs > 0:
        manifest = manifest[: args.limit_pairs]
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir / "poses")

    device = torch.device(args.device)
    flow_matcher = None
    if not args.dry_run:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --dry_run is set.")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        run_cfg = ckpt.get("run_config", {})
        state = ckpt["model_state"]
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            hidden_dim = run_cfg.get("hidden_dim", state["lig_emb.weight"].shape[0])
        n_layers = args.n_layers
        if n_layers is None:
            layer_indices = [int(k.split(".")[1]) for k in state.keys() if k.startswith("layers.")]
            n_layers = run_cfg.get("n_layers", max(layer_indices) + 1 if layer_indices else 6)
        model = build_default_model(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        flow_matcher = FlowMatcher(model, n_steps=args.n_steps).to(device)

    rows: list[dict[str, object]] = []
    for row in manifest:
        out = _run_one(
            flow_matcher=flow_matcher,  # type: ignore[arg-type]
            row=row,
            out_dir=out_dir,
            device=device,
            n_steps=args.n_steps,
            pocket_radius=args.pocket_radius,
            dry_run=args.dry_run,
        )
        rows.append(out)
        print(f"{out['pair_id']}: {out['status']} rmsd={out.get('rmsd')}")

    valid = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("rmsd"), (float, int)) and math.isfinite(float(r["rmsd"]))]
    success_top1 = sum(1 for r in valid if float(r["rmsd"]) < 2.0)
    summary = {
        "engine": "model",
        "n_total_pairs": len(rows),
        "n_scored_pairs": len(valid),
        "top1_rmsd_lt_2a_pct": round((success_top1 / max(1, len(valid))) * 100.0, 3),
        "dry_run": args.dry_run,
    }
    write_jsonl(out_dir / "model_results.jsonl", rows)
    write_json(out_dir / "model_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
