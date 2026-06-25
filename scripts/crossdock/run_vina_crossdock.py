import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.crossdock.common import (
    copy_pose_coords_onto_template,
    ensure_dir,
    load_first_mol,
    mol_coords,
    write_json,
    write_jsonl,
    write_pose_sdf,
)
from src.training.metrics import kabsch_rmsd


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _calc_center(sdf_path: Path) -> np.ndarray:
    mol = load_first_mol(sdf_path)
    return np.mean(mol_coords(mol), axis=0)


def _vina_to_sdf(vina_out_pdbqt: Path, template_sdf: Path, out_sdf: Path) -> None:
    # Requires OpenBabel. Writes Vina pose onto input ligand topology for RMSD atom ordering.
    cmd = ["obabel", str(vina_out_pdbqt), "-O", str(out_sdf)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"obabel conversion failed: {proc.stderr[-300:]}")
    converted = load_first_mol(out_sdf)
    template = load_first_mol(template_sdf)
    merged = copy_pose_coords_onto_template(template, converted)
    write_pose_sdf(merged, mol_coords(merged), out_sdf)


def _run_one(
    row: dict[str, str],
    out_dir: Path,
    box_size: float,
    exhaustiveness: int,
    dry_run: bool,
) -> dict[str, object]:
    pair_id = row["pair_id"]
    result: dict[str, object] = {"pair_id": pair_id, "target_id": row["target_id"], "engine": "vina"}
    ref_sdf = Path(row["reference_ligand_sdf"])
    lig_sdf = Path(row["ligand_input_sdf"])

    try:
        ref_mol = load_first_mol(ref_sdf)
        ref_coords = torch.tensor(mol_coords(ref_mol), dtype=torch.float32)

        if dry_run:
            pred_mol = load_first_mol(lig_sdf)
            pred_coords = torch.tensor(mol_coords(pred_mol), dtype=torch.float32)
            rmsd = float(kabsch_rmsd(pred_coords, ref_coords))
            result.update({"status": "ok", "rmsd": rmsd, "pred_pose_sdf": str(lig_sdf)})
            return result

        receptor_pdbqt = row.get("receptor_pdbqt", "")
        ligand_pdbqt = row.get("ligand_pdbqt", "")
        if not receptor_pdbqt or not ligand_pdbqt:
            raise RuntimeError("manifest requires receptor_pdbqt and ligand_pdbqt for vina runs")

        center = _calc_center(Path(row["receptor_native_ligand_sdf"]))
        out_pdbqt = out_dir / "poses" / f"{pair_id}_vina_out.pdbqt"
        cmd = [
            "vina",
            "--receptor",
            receptor_pdbqt,
            "--ligand",
            ligand_pdbqt,
            "--center_x",
            f"{center[0]:.4f}",
            "--center_y",
            f"{center[1]:.4f}",
            "--center_z",
            f"{center[2]:.4f}",
            "--size_x",
            f"{box_size:.3f}",
            "--size_y",
            f"{box_size:.3f}",
            "--size_z",
            f"{box_size:.3f}",
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            "1",
            "--out",
            str(out_pdbqt),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-500:] or "vina_failed")

        out_sdf = out_dir / "poses" / f"{pair_id}_vina_pred.sdf"
        _vina_to_sdf(out_pdbqt, lig_sdf, out_sdf)
        pred_mol = load_first_mol(out_sdf)
        rmsd = float(
            kabsch_rmsd(
                torch.tensor(mol_coords(pred_mol), dtype=torch.float32),
                ref_coords,
            )
        )
        result.update({"status": "ok", "rmsd": rmsd, "pred_pose_sdf": str(out_sdf)})
    except Exception as exc:
        result.update({"status": "failed", "rmsd": None, "error": str(exc)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vina CASF cross-docking.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--box_size", type=float, default=20.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--limit_pairs", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = _read_manifest(Path(args.manifest_csv).resolve())
    if args.limit_pairs > 0:
        manifest = manifest[: args.limit_pairs]
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir / "poses")

    rows: list[dict[str, object]] = []
    for row in manifest:
        out = _run_one(
            row=row,
            out_dir=out_dir,
            box_size=args.box_size,
            exhaustiveness=args.exhaustiveness,
            dry_run=args.dry_run,
        )
        rows.append(out)
        print(f"{out['pair_id']}: {out['status']} rmsd={out.get('rmsd')}")

    valid = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("rmsd"), (float, int)) and math.isfinite(float(r["rmsd"]))]
    success_top1 = sum(1 for r in valid if float(r["rmsd"]) < 2.0)
    summary = {
        "engine": "vina",
        "n_total_pairs": len(rows),
        "n_scored_pairs": len(valid),
        "top1_rmsd_lt_2a_pct": round((success_top1 / max(1, len(valid))) * 100.0, 3),
        "dry_run": args.dry_run,
    }
    write_jsonl(out_dir / "vina_results.jsonl", rows)
    write_json(out_dir / "vina_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
