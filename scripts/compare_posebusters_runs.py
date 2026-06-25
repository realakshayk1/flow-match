import argparse
import json
from pathlib import Path

import pandas as pd


META_COLS = {"complex_id", "rmsd", "shape_rmsd", "dock_rmsd",
             "uff_postprocess", "uff_failed", "pb_error", "pb_valid"}


def _bool_check_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def _pb_valid_rate(df: pd.DataFrame, checks: list[str]) -> tuple[int, float]:
    if not checks:
        return 0, 0.0
    valid = df[checks].all(axis=1)
    n_valid = int(valid.sum())
    pct = float(valid.mean() * 100.0)
    return n_valid, pct


def _safe_rmsd_stats(df: pd.DataFrame) -> dict:
    if "rmsd" not in df.columns or len(df) == 0:
        return {"rmsd_median": None, "rmsd_pct_under_2A": None}
    rmsd = pd.to_numeric(df["rmsd"], errors="coerce")
    rmsd = rmsd.dropna()
    if len(rmsd) == 0:
        return {"rmsd_median": None, "rmsd_pct_under_2A": None}
    return {
        "rmsd_median": round(float(rmsd.median()), 3),
        "rmsd_pct_under_2A": round(float((rmsd < 2.0).mean() * 100.0), 1),
    }


def _per_check_rate(df: pd.DataFrame, checks: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in checks:
        out[c] = round(float(df[c].mean() * 100.0), 1)
    return out


def _joint_valid_pct(df: pd.DataFrame, checks: list[str]) -> float | None:
    """Fraction with in-frame dock RMSD < 2A AND PB-valid (passes all checks)."""
    if not checks or "dock_rmsd" not in df.columns or len(df) == 0:
        return None
    dock = pd.to_numeric(df["dock_rmsd"], errors="coerce")
    valid = df[checks].all(axis=1)
    joint = (dock < 2.0) & valid
    return round(float(joint.mean() * 100.0), 1)


def _build_summary(df: pd.DataFrame) -> dict:
    checks = _bool_check_columns(df)
    n_valid, pb_valid_pct = _pb_valid_rate(df, checks)
    rmsd_stats = _safe_rmsd_stats(df)   # "rmsd" column is the in-frame dock RMSD
    pb_errors = int(df["pb_error"].notna().sum()) if "pb_error" in df.columns else 0
    return {
        "n_total": int(len(df)),
        "n_pb_valid": n_valid,
        "pb_valid_pct": round(pb_valid_pct, 1),
        "n_pb_error_rows": pb_errors,
        "n_check_columns": len(checks),
        "rmsd_median": rmsd_stats["rmsd_median"],
        "rmsd_pct_under_2A": rmsd_stats["rmsd_pct_under_2A"],
        "rmsd_lt2_and_pbvalid_pct": _joint_valid_pct(df, checks),
        "per_check_pass_rate": _per_check_rate(df, checks),
    }


def _to_markdown(report: dict) -> str:
    raw = report["raw"]
    uff = report["uff"]
    deltas = report["delta"]
    lines = [
        "# Phase 1 PoseBusters Report",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Raw | +UFF | Delta (+UFF - Raw) |",
        "|---|---:|---:|---:|",
        f"| N total | {raw['n_total']} | {uff['n_total']} | {deltas['n_total']} |",
        f"| PB-valid % | {raw['pb_valid_pct']} | {uff['pb_valid_pct']} | {deltas['pb_valid_pct']} |",
        f"| Dock RMSD median (A) | {raw['rmsd_median']} | {uff['rmsd_median']} | {deltas['rmsd_median']} |",
        f"| Dock RMSD < 2A (%) | {raw['rmsd_pct_under_2A']} | {uff['rmsd_pct_under_2A']} | {deltas['rmsd_pct_under_2A']} |",
        f"| RMSD<2A AND PB-valid (%) | {raw['rmsd_lt2_and_pbvalid_pct']} | {uff['rmsd_lt2_and_pbvalid_pct']} | {deltas['rmsd_lt2_and_pbvalid_pct']} |",
        f"| PB error rows | {raw['n_pb_error_rows']} | {uff['n_pb_error_rows']} | {deltas['n_pb_error_rows']} |",
        "",
        "## Data Quality Notes",
        "",
        f"- Raw run bool check columns detected: {raw['n_check_columns']}",
        f"- UFF run bool check columns detected: {uff['n_check_columns']}",
        "- If raw bool check columns are zero, that run likely failed due to PoseBusters API mismatch and should be rerun.",
        "",
        "## Top Per-Check Deltas",
        "",
    ]
    top_deltas = report["per_check_delta_top"]
    if not top_deltas:
        lines.append("- No overlapping check columns were found between runs.")
    else:
        lines.append("| Check | Raw % | +UFF % | Delta |")
        lines.append("|---|---:|---:|---:|")
        for row in top_deltas:
            lines.append(
                f"| {row['check']} | {row['raw_pct']} | {row['uff_pct']} | {row['delta_pct']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw vs UFF PoseBusters runs.")
    parser.add_argument("--raw", required=True, help="Path to raw results_raw.csv")
    parser.add_argument("--uff", required=True, help="Path to UFF results_raw.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw)
    uff_df = pd.read_csv(args.uff)

    raw_summary = _build_summary(raw_df)
    uff_summary = _build_summary(uff_df)

    delta = {}
    for k in ("n_total", "pb_valid_pct", "rmsd_median", "rmsd_pct_under_2A",
              "rmsd_lt2_and_pbvalid_pct", "n_pb_error_rows"):
        raw_v = raw_summary.get(k)
        uff_v = uff_summary.get(k)
        if raw_v is None or uff_v is None:
            delta[k] = None
        else:
            delta[k] = round(float(uff_v) - float(raw_v), 3)

    check_overlap = sorted(
        set(raw_summary["per_check_pass_rate"].keys()) & set(uff_summary["per_check_pass_rate"].keys())
    )
    per_check_delta = []
    for c in check_overlap:
        r = raw_summary["per_check_pass_rate"][c]
        u = uff_summary["per_check_pass_rate"][c]
        per_check_delta.append(
            {"check": c, "raw_pct": r, "uff_pct": u, "delta_pct": round(u - r, 1)}
        )
    per_check_delta_sorted = sorted(per_check_delta, key=lambda x: abs(x["delta_pct"]), reverse=True)

    report = {
        "raw": raw_summary,
        "uff": uff_summary,
        "delta": delta,
        "per_check_delta": per_check_delta_sorted,
        "per_check_delta_top": per_check_delta_sorted[:12],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(per_check_delta_sorted).to_csv(out_dir / "per_check_delta.csv", index=False)
    (out_dir / "phase1_summary.md").write_text(_to_markdown(report), encoding="utf-8")

    print(f"Wrote report to {out_dir}")


if __name__ == "__main__":
    main()
