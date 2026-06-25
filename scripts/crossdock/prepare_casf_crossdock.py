import argparse
import csv
from collections import defaultdict
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.crossdock.common import ensure_dir


def _load_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"complex_id", "target_id", "receptor_pdb", "ligand_sdf"}
    missing = required.difference(set(rows[0].keys()) if rows else set())
    if missing:
        raise ValueError(
            "CASF index missing columns: "
            + ", ".join(sorted(missing))
            + ". Required: complex_id,target_id,receptor_pdb,ligand_sdf"
        )
    return rows


def _resolve(path_value: str, root_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (root_dir / p).resolve()


def build_manifest(
    index_rows: list[dict[str, str]],
    root_dir: Path,
    allow_self_dock: bool,
    limit_pairs: int | None,
) -> list[dict[str, str]]:
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        by_target[row["target_id"]].append(row)

    manifest: list[dict[str, str]] = []
    for target_id, members in sorted(by_target.items()):
        for receptor_row in members:
            receptor_complex_id = receptor_row["complex_id"]
            receptor_pdb = _resolve(receptor_row["receptor_pdb"], root_dir)
            receptor_native = _resolve(receptor_row["ligand_sdf"], root_dir)
            for ligand_row in members:
                ligand_complex_id = ligand_row["complex_id"]
                if not allow_self_dock and ligand_complex_id == receptor_complex_id:
                    continue
                ligand_sdf = _resolve(ligand_row["ligand_sdf"], root_dir)
                pair_id = f"{target_id}__R-{receptor_complex_id}__L-{ligand_complex_id}"
                manifest.append(
                    {
                        "pair_id": pair_id,
                        "target_id": target_id,
                        "receptor_complex_id": receptor_complex_id,
                        "ligand_complex_id": ligand_complex_id,
                        "receptor_pdb": str(receptor_pdb),
                        "receptor_native_ligand_sdf": str(receptor_native),
                        "ligand_input_sdf": str(ligand_sdf),
                        # Protocol assumption: this must already be in receptor frame.
                        "reference_ligand_sdf": str(ligand_sdf),
                    }
                )
                if limit_pairs and len(manifest) >= limit_pairs:
                    return manifest
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CASF-2016 cross-docking manifest.")
    parser.add_argument("--index_csv", required=True, help="CASF index with target and path columns.")
    parser.add_argument("--root_dir", default=".", help="Base directory for relative paths in index_csv.")
    parser.add_argument("--out_manifest", required=True, help="Output CSV path.")
    parser.add_argument("--allow_self_dock", action="store_true", help="Include receptor=ligand source pairs.")
    parser.add_argument("--limit_pairs", type=int, default=0, help="Optional cap for smoke runs.")
    args = parser.parse_args()

    index_csv = Path(args.index_csv).resolve()
    root_dir = Path(args.root_dir).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    ensure_dir(out_manifest.parent)

    rows = _load_rows(index_csv)
    manifest_rows = build_manifest(
        index_rows=rows,
        root_dir=root_dir,
        allow_self_dock=args.allow_self_dock,
        limit_pairs=args.limit_pairs if args.limit_pairs > 0 else None,
    )
    if not manifest_rows:
        raise SystemExit("No cross-docking pairs created; check target grouping and flags.")

    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} pairs to {out_manifest}")


if __name__ == "__main__":
    main()
