import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.crossdock.common import ensure_dir, read_jsonl, write_json


def _score(rows: list[dict], threshold: float) -> dict:
    ok = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("rmsd"), (float, int))]
    n_ok = len(ok)
    n_success = sum(1 for r in ok if float(r["rmsd"]) < threshold)
    return {
        "n_total_pairs": len(rows),
        "n_scored_pairs": n_ok,
        "top1_rmsd_lt_2a_count": n_success,
        "top1_rmsd_lt_2a_pct": round((n_success / max(1, n_ok)) * 100.0, 3),
    }


def _per_target(rows: list[dict], threshold: float) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("target_id", "unknown")), []).append(row)
    return {target: _score(items, threshold) for target, items in sorted(grouped.items())}


def _common_pairs(model_rows: list[dict], vina_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    model_map = {str(r["pair_id"]): r for r in model_rows if "pair_id" in r}
    vina_map = {str(r["pair_id"]): r for r in vina_rows if "pair_id" in r}
    common = sorted(set(model_map).intersection(vina_map))
    return [model_map[p] for p in common], [vina_map[p] for p in common]


def _render_md(report: dict) -> str:
    lines = [
        "# Phase 3 Cross-Docking Report",
        "",
        "## Overall",
        "",
        f"- Common comparable pairs: {report['n_common_pairs']}",
        f"- Model top-1 RMSD<2A: {report['model']['top1_rmsd_lt_2a_pct']}% ({report['model']['top1_rmsd_lt_2a_count']}/{report['model']['n_scored_pairs']})",
        f"- Vina top-1 RMSD<2A: {report['vina']['top1_rmsd_lt_2a_pct']}% ({report['vina']['top1_rmsd_lt_2a_count']}/{report['vina']['n_scored_pairs']})",
        "",
        "## Per Target (Common Pairs)",
        "",
    ]
    targets = sorted(set(report["per_target_model"]).union(report["per_target_vina"]))
    for target in targets:
        m = report["per_target_model"].get(target, {})
        v = report["per_target_vina"].get(target, {})
        lines.append(
            f"- {target}: model={m.get('top1_rmsd_lt_2a_pct', 'n/a')}% "
            f"({m.get('top1_rmsd_lt_2a_count', 0)}/{m.get('n_scored_pairs', 0)}), "
            f"vina={v.get('top1_rmsd_lt_2a_pct', 'n/a')}% "
            f"({v.get('top1_rmsd_lt_2a_count', 0)}/{v.get('n_scored_pairs', 0)})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model and Vina cross-docking outputs.")
    parser.add_argument("--model_results", required=True)
    parser.add_argument("--vina_results", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    model_rows = read_jsonl(Path(args.model_results).resolve())
    vina_rows = read_jsonl(Path(args.vina_results).resolve())
    model_common, vina_common = _common_pairs(model_rows, vina_rows)

    report = {
        "n_common_pairs": len(model_common),
        "threshold_angstrom": args.threshold,
        "model": _score(model_common, args.threshold),
        "vina": _score(vina_common, args.threshold),
        "per_target_model": _per_target(model_common, args.threshold),
        "per_target_vina": _per_target(vina_common, args.threshold),
    }
    write_json(out_dir / "crossdock_comparison.json", report)
    (out_dir / "crossdock_report.md").write_text(_render_md(report), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
