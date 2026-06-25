"""
Run full Phase 1 PoseBusters pipeline: raw eval, +UFF eval, comparison report, optional baseline bundle.

Example:
  python scripts/run_phase1_bundle.py \\
    --checkpoint checkpoints/best_model.pt \\
    --out_root results/phase1 \\
    --split test \\
    --n_inference_steps 20 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 1: PoseBusters raw + UFF + comparison + bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_inference_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_complexes", type=int, default=None)
    parser.add_argument("--out_root", default="results/phase1")
    parser.add_argument(
        "--relax", choices=["uff", "pocket"], default="pocket",
        help="Relaxation mode for the second (post-processed) run; raw run is always --relax none.",
    )
    parser.add_argument(
        "--skip_raw",
        action="store_true",
        help="Skip raw eval (reuse existing posebusters_raw under out_root).",
    )
    parser.add_argument(
        "--skip_uff",
        action="store_true",
        help="Skip UFF eval (reuse existing posebusters_uff under out_root).",
    )
    args = parser.parse_args()

    py = sys.executable
    out_root = Path(args.out_root).resolve()
    raw_dir = out_root / "posebusters_raw"
    uff_dir = out_root / "posebusters_uff"
    compare_dir = out_root / "comparison"
    bundle_dir = out_root / "baseline_bundle"

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)

    common = [
        py,
        str(root / "scripts" / "eval_posebusters.py"),
        "--checkpoint",
        args.checkpoint,
        "--processed_dir",
        args.processed_dir,
        "--splits",
        args.splits,
        "--split",
        args.split,
        "--n_inference_steps",
        str(args.n_inference_steps),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.max_complexes is not None:
        common.extend(["--max_complexes", str(args.max_complexes)])
    common.extend(["--output_dir"])

    if not args.skip_raw:
        run(common + [str(raw_dir), "--relax", "none"])

    if not args.skip_uff:
        run(common + [str(uff_dir), "--relax", args.relax])

    run(
        [
            py,
            str(root / "scripts" / "compare_posebusters_runs.py"),
            "--raw",
            str(raw_dir / "results_raw.csv"),
            "--uff",
            str(uff_dir / "results_raw.csv"),
            "--out_dir",
            str(compare_dir),
        ]
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compare_dir / "phase1_summary.json", bundle_dir / "phase1_comparison.json")
    shutil.copy2(compare_dir / "phase1_summary.md", bundle_dir / "phase1_comparison.md")
    if raw_dir.joinpath("results_summary.json").is_file():
        shutil.copy2(raw_dir / "results_summary.json", bundle_dir / "summary_raw.json")
    if raw_dir.joinpath("results_raw.csv").is_file():
        shutil.copy2(raw_dir / "results_raw.csv", bundle_dir / "per_complex_raw.csv")
    if uff_dir.joinpath("results_summary.json").is_file():
        shutil.copy2(uff_dir / "results_summary.json", bundle_dir / "summary_uff.json")
    if uff_dir.joinpath("results_raw.csv").is_file():
        shutil.copy2(uff_dir / "results_raw.csv", bundle_dir / "per_complex_uff.csv")

    print(f"\nPhase 1 artifacts:\n  {compare_dir}\n  {bundle_dir}")


if __name__ == "__main__":
    main()
