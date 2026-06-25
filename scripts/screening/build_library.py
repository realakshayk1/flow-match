import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class FilterConfig:
    min_mw: float = 180.0
    max_mw: float = 550.0
    max_logp: float = 5.5
    max_hbd: int = 5
    max_hba: int = 10
    max_tpsa: float = 150.0
    max_rot_bonds: int = 10
    min_heavy_atoms: int = 10


def _standardize_smiles(smiles: str) -> tuple[str | None, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    try:
        fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        uncharger = rdMolStandardize.Uncharger()
        mol = fragment_chooser.choose(mol)
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        std = Chem.MolToSmiles(mol, canonical=True)
        return std, ""
    except Exception:
        return None, "standardization_failed"


def _descriptor_row(mol: Chem.Mol) -> dict[str, float]:
    return {
        "mw": round(float(Descriptors.MolWt(mol)), 3),
        "logp": round(float(Crippen.MolLogP(mol)), 3),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "tpsa": round(float(Descriptors.TPSA(mol)), 3),
        "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def _passes_filters(desc: dict[str, float], config: FilterConfig) -> tuple[bool, str]:
    checks = [
        (desc["mw"] >= config.min_mw, "mw_below_min"),
        (desc["mw"] <= config.max_mw, "mw_above_max"),
        (desc["logp"] <= config.max_logp, "logp_above_max"),
        (desc["hbd"] <= config.max_hbd, "hbd_above_max"),
        (desc["hba"] <= config.max_hba, "hba_above_max"),
        (desc["tpsa"] <= config.max_tpsa, "tpsa_above_max"),
        (desc["rot_bonds"] <= config.max_rot_bonds, "rot_bonds_above_max"),
        (desc["heavy_atoms"] >= config.min_heavy_atoms, "too_few_heavy_atoms"),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and clean a screening library. "
            "Input contract: CSV with columns compound_id, smiles[, source]. "
            "Output contract: cleaned CSV with standardized SMILES + descriptors, "
            "filter_log CSV, and stats JSON."
        )
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--filter_log_csv", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    rows = _read_rows(input_path)
    if args.max_records is not None:
        rows = rows[: max(0, args.max_records)]

    config = FilterConfig()
    seen_std_smiles: set[str] = set()
    accepted_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        compound_id = row.get("compound_id") or f"cmpd_{idx}"
        smiles = (row.get("smiles") or "").strip()
        source = row.get("source", "")
        std_smiles, err = _standardize_smiles(smiles)
        if std_smiles is None:
            log_rows.append(
                {
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "status": "rejected",
                    "reason": err,
                }
            )
            continue
        if std_smiles in seen_std_smiles:
            log_rows.append(
                {
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "status": "rejected",
                    "reason": "duplicate_standardized_smiles",
                }
            )
            continue

        mol = Chem.MolFromSmiles(std_smiles)
        if mol is None:
            log_rows.append(
                {
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "status": "rejected",
                    "reason": "invalid_after_standardization",
                }
            )
            continue

        desc = _descriptor_row(mol)
        keep, reason = _passes_filters(desc, config)
        if not keep:
            log_rows.append(
                {
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "status": "rejected",
                    "reason": reason,
                }
            )
            continue

        seen_std_smiles.add(std_smiles)
        accepted_rows.append(
            {
                "compound_id": compound_id,
                "source": source,
                "input_smiles": smiles,
                "std_smiles": std_smiles,
                **desc,
            }
        )
        log_rows.append(
            {
                "compound_id": compound_id,
                "smiles": smiles,
                "status": "accepted",
                "reason": "",
            }
        )

    output_csv = Path(args.output_csv)
    filter_log_csv = Path(args.filter_log_csv)
    stats_json = Path(args.stats_json)

    _write_csv(
        output_csv,
        accepted_rows,
        fieldnames=[
            "compound_id",
            "source",
            "input_smiles",
            "std_smiles",
            "mw",
            "logp",
            "hba",
            "hbd",
            "tpsa",
            "rot_bonds",
            "heavy_atoms",
        ],
    )
    _write_csv(filter_log_csv, log_rows, fieldnames=["compound_id", "smiles", "status", "reason"])

    n_in = len(rows)
    n_out = len(accepted_rows)
    stats = {
        "input_csv": str(input_path.as_posix()),
        "output_csv": str(output_csv.as_posix()),
        "n_input": n_in,
        "n_kept": n_out,
        "n_rejected": n_in - n_out,
        "keep_rate_pct": round((n_out / max(1, n_in)) * 100.0, 2),
        "target_filters": {
            "mw_range": [config.min_mw, config.max_mw],
            "max_logp": config.max_logp,
            "max_hbd": config.max_hbd,
            "max_hba": config.max_hba,
            "max_tpsa": config.max_tpsa,
            "max_rot_bonds": config.max_rot_bonds,
            "min_heavy_atoms": config.min_heavy_atoms,
        },
        "descriptor_means": {
            "mw": round(sum(r["mw"] for r in accepted_rows) / max(1, n_out), 3) if n_out else math.nan,
            "logp": round(sum(r["logp"] for r in accepted_rows) / max(1, n_out), 3) if n_out else math.nan,
            "tpsa": round(sum(r["tpsa"] for r in accepted_rows) / max(1, n_out), 3) if n_out else math.nan,
        },
    }
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
