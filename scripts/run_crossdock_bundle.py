"""
Orchestrate Phase 3 cross-docking: optional PDBQT prep, model run, Vina run, comparison.

Smoke (no checkpoint, no Vina):
  python scripts/run_crossdock_bundle.py \\
    --manifest_csv eval/crossdock/manifest.csv \\
    --out_root results/phase3/run_smoke \\
    --dry_run

Full run requires checkpoint, obabel (`prepare_pdbqt_for_vina`), and `vina` on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    parser = argparse.ArgumentParser(description="Phase 3 cross-docking bundle runner.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_root", default="results/phase3/run")
    parser.add_argument("--checkpoint", default="", help="Flow-Match checkpoint (required unless --dry_run).")
    parser.add_argument("--pdbqt_cache_dir", default="eval/pdbqt_cache")
    parser.add_argument("--skip_pdbqt", action="store_true", help="Manifest already has receptor_pdbqt/ligand_pdbqt.")
    parser.add_argument("--dry_run", action="store_true", help="Model and Vina use dry_run modes.")
    parser.add_argument("--limit_pairs", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    model_dir = out_root / "model"
    vina_dir = out_root / "vina"
    cmp_dir = out_root / "compare"
    manifest = Path(args.manifest_csv).resolve()

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)

    manifest_run = manifest
    if not args.skip_pdbqt and not args.dry_run:
        enriched = out_root / "manifest_with_pdbqt.csv"
        run(
            [
                py,
                str(root / "scripts" / "prepare_pdbqt_for_vina.py"),
                "--input_csv",
                str(manifest),
                "--out_csv",
                str(enriched),
                "--cache_dir",
                str(Path(args.pdbqt_cache_dir).resolve()),
                "--skip_existing",
            ]
        )
        manifest_run = enriched

    model_cmd = [
        py,
        str(root / "scripts" / "crossdock" / "run_model_crossdock.py"),
        "--manifest_csv",
        str(manifest_run),
        "--out_dir",
        str(model_dir),
        "--n_steps",
        str(args.n_steps),
        "--device",
        args.device,
    ]
    if args.limit_pairs:
        model_cmd.extend(["--limit_pairs", str(args.limit_pairs)])
    if args.dry_run:
        model_cmd.append("--dry_run")
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --dry_run")
        model_cmd.extend(["--checkpoint", args.checkpoint])
    run(model_cmd)

    vina_cmd = [
        py,
        str(root / "scripts" / "crossdock" / "run_vina_crossdock.py"),
        "--manifest_csv",
        str(manifest_run),
        "--out_dir",
        str(vina_dir),
    ]
    if args.limit_pairs:
        vina_cmd.extend(["--limit_pairs", str(args.limit_pairs)])
    if args.dry_run:
        vina_cmd.append("--dry_run")
    run(vina_cmd)

    run(
        [
            py,
            str(root / "scripts" / "crossdock" / "evaluate_crossdock.py"),
            "--model_results",
            str(model_dir / "model_results.jsonl"),
            "--vina_results",
            str(vina_dir / "vina_results.jsonl"),
            "--out_dir",
            str(cmp_dir),
        ]
    )

    shutil.copy2(cmp_dir / "crossdock_report.md", out_root / "model_vs_vina.md")
    print(f"\nPhase 3 outputs under {out_root}")


if __name__ == "__main__":
    main()
