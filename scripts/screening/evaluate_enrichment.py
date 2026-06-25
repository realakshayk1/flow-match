import argparse
import csv
import json
import math
from pathlib import Path

from rdkit import Chem


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def _ef_at_percent(rows: list[dict[str, object]], pct: float) -> tuple[float, int, int]:
    n_total = len(rows)
    n_actives = sum(1 for r in rows if bool(r["is_active"]))
    if n_total == 0 or n_actives == 0:
        return 0.0, 0, 0
    top_k = max(1, int(math.ceil(n_total * pct)))
    top_hits = rows[:top_k]
    top_actives = sum(1 for r in top_hits if bool(r["is_active"]))
    observed = top_actives / top_k
    expected = n_actives / n_total
    return observed / expected, top_k, top_actives


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute enrichment metrics for ranked screening output. "
            "Input contract: ranked CSV with rank,compound_id,std_smiles. "
            "Known actives CSV contract: target,compound_id,std_smiles,is_active. "
            "Output contract: enrichment JSON and per-compound annotated CSV."
        )
    )
    parser.add_argument("--ranked_csv", required=True)
    parser.add_argument("--known_actives_csv", required=True)
    parser.add_argument("--target", choices=["cdk2", "egfr"], default="cdk2")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_annotated_csv", required=True)
    args = parser.parse_args()

    ranked = _read_csv(Path(args.ranked_csv))
    known = _read_csv(Path(args.known_actives_csv))
    if not ranked:
        raise ValueError("ranked_csv has no rows")

    active_ids: set[str] = set()
    active_smiles: set[str] = set()
    for row in known:
        if row.get("target", "").lower() != args.target:
            continue
        if str(row.get("is_active", "1")).strip() not in {"1", "true", "True", "yes", "YES"}:
            continue
        if row.get("compound_id"):
            active_ids.add(row["compound_id"])
        if row.get("std_smiles"):
            active_smiles.add(_canon(row["std_smiles"]))

    annotated: list[dict[str, object]] = []
    for row in sorted(ranked, key=lambda r: int(r.get("rank", "999999"))):
        matched_by = ""
        cid = row.get("compound_id", "")
        if cid in active_ids:
            matched_by = "compound_id"
        else:
            smi = _canon(row.get("std_smiles", ""))
            if smi and smi in active_smiles:
                matched_by = "std_smiles"
        annotated.append(
            {
                **row,
                "is_active": bool(matched_by),
                "matched_by": matched_by,
            }
        )

    ef1, top_k, top_actives = _ef_at_percent(annotated, 0.01)
    total_actives = sum(1 for r in annotated if bool(r["is_active"]))
    n_total = len(annotated)
    enrichment = {
        "target": args.target,
        "ranked_csv": args.ranked_csv,
        "known_actives_csv": args.known_actives_csv,
        "n_total_ranked": n_total,
        "n_total_actives": total_actives,
        "ef_1pct": round(float(ef1), 4),
        "top_1pct_k": top_k,
        "actives_in_top_1pct": top_actives,
        "active_rate_overall": round(total_actives / max(1, n_total), 4),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(enrichment, indent=2), encoding="utf-8")
    _write_csv(
        Path(args.out_annotated_csv),
        annotated,
        fieldnames=[
            "rank",
            "compound_id",
            "target",
            "predicted_score_kcal_mol",
            "score_source",
            "std_smiles",
            "is_active",
            "matched_by",
        ],
    )
    print(json.dumps(enrichment, indent=2))


if __name__ == "__main__":
    main()
