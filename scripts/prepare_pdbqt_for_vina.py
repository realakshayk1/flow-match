"""
Build AutoDock Vina PDBQT inputs via Open Babel with on-disk caching.

Typical flows:
  - Cross-dock manifest from `prepare_casf_crossdock.py` (receptor_pdb, ligand_input_sdf)
  - PDBBind throughput manifest (protein_path, ligand_path)

Usage:
  python scripts/prepare_pdbqt_for_vina.py \\
    --input_csv eval/crossdock/manifest.csv \\
    --out_csv eval/crossdock/manifest_with_pdbqt.csv \\
    --cache_dir eval/pdbqt_cache

Requires `obabel` on PATH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _resolve_paths(row: dict[str, str]) -> tuple[str | None, str | None]:
    rec = (
        row.get("receptor_pdb")
        or row.get("protein_path")
        or row.get("receptor_pdb_path")
        or ""
    ).strip()
    lig = (
        row.get("ligand_input_sdf")
        or row.get("ligand_sdf")
        or row.get("ligand_path")
        or ""
    ).strip()
    return (rec or None, lig or None)


def _cache_key(receptor: Path, ligand: Path) -> str:
    payload = f"{receptor.resolve()}|{ligand.resolve()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_obabel(inp: Path, out: Path, extra: list[str]) -> tuple[int, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["obabel", str(inp), "-O", str(out)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stderr or "")[-400:]
    return proc.returncode, tail


def _receptor_to_pdbqt(src: Path, dst: Path) -> tuple[bool, str]:
    # Rigid receptor root (-xr) is standard for docking pipelines using obabel.
    rc, err = _run_obabel(src, dst, ["-xr"])
    if rc != 0:
        rc2, err2 = _run_obabel(src, dst, [])
        if rc2 != 0:
            return False, err or err2
    return True, ""


def _ligand_to_pdbqt(src: Path, dst: Path) -> tuple[bool, str]:
    rc, err = _run_obabel(src, dst, [])
    if rc != 0:
        return False, err
    return True, ""


def enrich_rows(
    rows: list[dict[str, str]],
    cache_dir: Path,
    skip_existing: bool,
) -> tuple[list[dict[str, Any]], int]:
    rec_dir = cache_dir / "receptors"
    lig_dir = cache_dir / "ligands"
    out_rows: list[dict[str, Any]] = []
    failures = 0
    for row in rows:
        new_row: dict[str, Any] = dict(row)
        rec_in_s, lig_in_s = _resolve_paths(row)
        new_row["pdbqt_error"] = ""
        if not rec_in_s or not lig_in_s:
            new_row["pdbqt_error"] = "missing_receptor_or_ligand_path"
            new_row["receptor_pdbqt"] = row.get("receptor_pdbqt", "")
            new_row["ligand_pdbqt"] = row.get("ligand_pdbqt", "")
            failures += 1
            out_rows.append(new_row)
            continue

        receptor = Path(rec_in_s)
        ligand = Path(lig_in_s)
        if not receptor.is_file() or not ligand.is_file():
            new_row["pdbqt_error"] = "path_not_found"
            new_row["receptor_pdbqt"] = row.get("receptor_pdbqt", "")
            new_row["ligand_pdbqt"] = row.get("ligand_pdbqt", "")
            failures += 1
            out_rows.append(new_row)
            continue

        key = _cache_key(receptor, ligand)
        rec_out = rec_dir / f"{key[:24]}.pdbqt"
        lig_out = lig_dir / f"{key[:24]}_lig.pdbqt"

        if not skip_existing or not rec_out.is_file():
            ok, err = _receptor_to_pdbqt(receptor, rec_out)
            if not ok:
                new_row["pdbqt_error"] = f"receptor_obabel:{err}"
                new_row["receptor_pdbqt"] = ""
                new_row["ligand_pdbqt"] = ""
                failures += 1
                out_rows.append(new_row)
                continue
        if not skip_existing or not lig_out.is_file():
            ok, err = _ligand_to_pdbqt(ligand, lig_out)
            if not ok:
                new_row["pdbqt_error"] = f"ligand_obabel:{err}"
                new_row["receptor_pdbqt"] = str(rec_out.resolve()) if rec_out.is_file() else ""
                new_row["ligand_pdbqt"] = ""
                failures += 1
                out_rows.append(new_row)
                continue

        new_row["receptor_pdbqt"] = str(rec_out.resolve())
        new_row["ligand_pdbqt"] = str(lig_out.resolve())
        out_rows.append(new_row)
    return out_rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PDBQT files for AutoDock Vina via Open Babel.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cache_dir", required=True, help="Directory for cached receptor/ligand PDBQT files.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse cached PDBQT if present without re-running obabel.",
    )
    args = parser.parse_args()
    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel binary `obabel` not found on PATH. Install Open Babel to prepare PDBQT files.")

    inp = Path(args.input_csv).resolve()
    out = Path(args.out_csv).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    with inp.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input_csv is empty")

    enriched, failures = enrich_rows(rows, cache_dir, skip_existing=args.skip_existing)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen_fn: set[str] = set()
    for row in enriched:
        for k in row:
            if k not in seen_fn:
                seen_fn.add(k)
                fieldnames.append(k)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Wrote {len(enriched)} rows to {out} ({failures} rows missing PDBQT)")


if __name__ == "__main__":
    main()
