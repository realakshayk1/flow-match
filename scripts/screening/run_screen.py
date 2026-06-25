import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _deterministic_noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return (bucket - 0.5) * 1.2


def _mock_score(row: dict[str, str], target: str) -> float:
    mw = float(row.get("mw", 0.0))
    logp = float(row.get("logp", 0.0))
    tpsa = float(row.get("tpsa", 0.0))
    hba = float(row.get("hba", 0.0))
    hbd = float(row.get("hbd", 0.0))
    rot_bonds = float(row.get("rot_bonds", 0.0))
    base = -7.2 if target == "cdk2" else -7.5
    penalty = (
        abs(logp - 2.2) * 0.35
        + abs(tpsa - 90.0) * 0.01
        + max(0.0, mw - 460.0) * 0.004
        + max(0.0, rot_bonds - 8.0) * 0.18
    )
    hbond_bonus = -0.12 * min(hba + hbd, 8.0)
    noise = _deterministic_noise(row.get("std_smiles", row.get("compound_id", "")))
    return round(base + penalty + hbond_bonus + noise, 4)


def _summary(per_ligand_ms: list[float], scored_rows: int) -> dict[str, float | int]:
    if not per_ligand_ms:
        return {
            "n_scored": scored_rows,
            "mean_ms_per_ligand": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }
    vals = sorted(per_ligand_ms)
    idx95 = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return {
        "n_scored": scored_rows,
        "mean_ms_per_ligand": round(float(statistics.mean(vals)), 4),
        "p50_ms": round(float(statistics.median(vals)), 4),
        "p95_ms": round(float(vals[idx95]), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Target-centric screen runner with runtime accounting. "
            "Input contract: cleaned library CSV from build_library.py containing "
            "compound_id,std_smiles and descriptor columns. "
            "Output contract: ranked CSV with score/rank and runtime JSON summary."
        )
    )
    parser.add_argument("--library_csv", required=True)
    parser.add_argument("--target", choices=["cdk2", "egfr"], default="cdk2")
    parser.add_argument("--out_ranked_csv", required=True)
    parser.add_argument("--out_runtime_json", required=True)
    parser.add_argument(
        "--precomputed_scores_csv",
        default=None,
        help="Optional CSV with compound_id and predicted_score_kcal_mol for external model scores.",
    )
    args = parser.parse_args()

    library_rows = _read_csv(Path(args.library_csv))
    if not library_rows:
        raise ValueError("library_csv has no rows")

    score_map: dict[str, float] = {}
    if args.precomputed_scores_csv:
        for row in _read_csv(Path(args.precomputed_scores_csv)):
            if row.get("compound_id"):
                score_map[row["compound_id"]] = float(row["predicted_score_kcal_mol"])

    scored: list[dict[str, object]] = []
    per_ligand_ms: list[float] = []
    t0_all = time.perf_counter()

    for row in library_rows:
        start = time.perf_counter()
        compound_id = row.get("compound_id", "")
        if compound_id in score_map:
            score = round(score_map[compound_id], 4)
            source = "precomputed"
        else:
            score = _mock_score(row, target=args.target)
            source = "mock_target_model"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_ligand_ms.append(elapsed_ms)

        scored.append(
            {
                "compound_id": compound_id,
                "std_smiles": row.get("std_smiles", ""),
                "predicted_score_kcal_mol": score,
                "score_source": source,
                "target": args.target,
            }
        )

    scored.sort(key=lambda r: float(r["predicted_score_kcal_mol"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    out_ranked = Path(args.out_ranked_csv)
    _write_csv(
        out_ranked,
        scored,
        fieldnames=[
            "rank",
            "compound_id",
            "target",
            "predicted_score_kcal_mol",
            "score_source",
            "std_smiles",
        ],
    )

    total_s = time.perf_counter() - t0_all
    runtime = {
        "target": args.target,
        "library_csv": args.library_csv,
        "ranked_csv": str(out_ranked.as_posix()),
        "scoring_mode": "precomputed+mock_fallback" if args.precomputed_scores_csv else "mock_target_model",
        "total_wall_time_s": round(total_s, 4),
        "ligands_per_s": round(len(scored) / max(total_s, 1e-9), 3),
        **_summary(per_ligand_ms, len(scored)),
    }
    out_runtime = Path(args.out_runtime_json)
    out_runtime.parent.mkdir(parents=True, exist_ok=True)
    out_runtime.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
