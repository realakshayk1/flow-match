"""
Phase 4 one-target screening bundle: clean library → rank → enrichment → top-hit report.

Example (CDK2 toy inputs):
  python scripts/run_phase4_screening_bundle.py \\
    --target cdk2 \\
    --input_csv data/screening/library_templates/toy_library.csv \\
    --known_actives_csv data/screening/targets/cdk2/known_actives.csv \\
    --out_root results/phase4/cdk2_bundle
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    parser = argparse.ArgumentParser(description="Phase 4 screening + EF1% + top-hit bundle.")
    parser.add_argument("--target", choices=["cdk2", "egfr"], default="cdk2")
    parser.add_argument("--input_csv", required=True, help="Raw library CSV for build_library.")
    parser.add_argument("--known_actives_csv", required=True)
    parser.add_argument("--out_root", default="results/phase4/bundle")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    clean_csv = out_root / "clean_library.csv"
    filter_log = out_root / "filter_log.csv"
    stats_json = out_root / "library_stats.json"
    ranked_csv = out_root / "ranked_hits.csv"
    runtime_json = out_root / "runtime.json"
    enrich_json = out_root / "enrichment_summary.json"
    annotated_csv = out_root / "ranked_annotated.csv"
    top_csv = out_root / "top_hits.csv"
    top_md = out_root / "top_hits_report.md"

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)

    out_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            py,
            str(root / "scripts" / "screening" / "build_library.py"),
            "--input_csv",
            str(Path(args.input_csv).resolve()),
            "--output_csv",
            str(clean_csv),
            "--filter_log_csv",
            str(filter_log),
            "--stats_json",
            str(stats_json),
        ]
    )

    run(
        [
            py,
            str(root / "scripts" / "screening" / "run_screen.py"),
            "--library_csv",
            str(clean_csv),
            "--target",
            args.target,
            "--out_ranked_csv",
            str(ranked_csv),
            "--out_runtime_json",
            str(runtime_json),
        ]
    )

    run(
        [
            py,
            str(root / "scripts" / "screening" / "evaluate_enrichment.py"),
            "--ranked_csv",
            str(ranked_csv),
            "--known_actives_csv",
            str(Path(args.known_actives_csv).resolve()),
            "--target",
            args.target,
            "--out_json",
            str(enrich_json),
            "--out_annotated_csv",
            str(annotated_csv),
        ]
    )

    run(
        [
            py,
            str(root / "scripts" / "screening" / "analyze_top_hits.py"),
            "--ranked_csv",
            str(ranked_csv),
            "--known_actives_csv",
            str(Path(args.known_actives_csv).resolve()),
            "--target",
            args.target,
            "--top_n",
            str(args.top_n),
            "--out_csv",
            str(top_csv),
            "--out_md",
            str(top_md),
        ]
    )

    summary_dst = out_root / "SCREENING_SUMMARY.txt"
    summary_dst.write_text(
        f"target={args.target}\n"
        f"enrichment_summary={enrich_json}\n"
        f"ranked_hits={ranked_csv}\n"
        f"runtime={runtime_json}\n"
        f"top_hits_report={top_md}\n",
        encoding="utf-8",
    )
    print(f"\nPhase 4 bundle complete: {out_root}")


if __name__ == "__main__":
    main()
