import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GEN.GetFingerprint(mol)


def _descriptor_pack(smiles: str) -> dict[str, float | str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mw": 0.0, "logp": 0.0, "tpsa": 0.0}
    return {
        "mw": round(float(Descriptors.MolWt(mol)), 3),
        "logp": round(float(Crippen.MolLogP(mol)), 3),
        "tpsa": round(float(Descriptors.TPSA(mol)), 3),
    }


def _novelty_bucket(max_tanimoto: float) -> str:
    if max_tanimoto < 0.3:
        return "high_novelty"
    if max_tanimoto < 0.6:
        return "moderate_novelty"
    return "low_novelty"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Top-hit interpretation scaffold. "
            "Input contract: ranked CSV + known actives CSV with std_smiles. "
            "Output contract: top-hit CSV and markdown report with novelty/descriptor summary."
        )
    )
    parser.add_argument("--ranked_csv", required=True)
    parser.add_argument("--known_actives_csv", required=True)
    parser.add_argument("--target", choices=["cdk2", "egfr"], default="cdk2")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    ranked = sorted(_read_csv(Path(args.ranked_csv)), key=lambda r: int(r.get("rank", "999999")))
    top_hits = ranked[: max(1, args.top_n)]
    known = [r for r in _read_csv(Path(args.known_actives_csv)) if r.get("target", "").lower() == args.target]

    known_fps = []
    for row in known:
        smi = row.get("std_smiles", "")
        fp = _morgan_fp(smi)
        if fp is not None:
            known_fps.append(fp)

    analyzed: list[dict[str, object]] = []
    for row in top_hits:
        smiles = row.get("std_smiles", "")
        hit_fp = _morgan_fp(smiles)
        max_tanimoto = 0.0
        if hit_fp is not None and known_fps:
            max_tanimoto = max(TanimotoSimilarity(hit_fp, ref_fp) for ref_fp in known_fps)

        descriptors = _descriptor_pack(smiles)
        analyzed.append(
            {
                "rank": row.get("rank", ""),
                "compound_id": row.get("compound_id", ""),
                "predicted_score_kcal_mol": row.get("predicted_score_kcal_mol", ""),
                "std_smiles": smiles,
                "max_tanimoto_to_known_active": round(float(max_tanimoto), 4),
                "novelty_bucket": _novelty_bucket(max_tanimoto),
                **descriptors,
            }
        )

    out_csv = Path(args.out_csv)
    _write_csv(
        out_csv,
        analyzed,
        fieldnames=[
            "rank",
            "compound_id",
            "predicted_score_kcal_mol",
            "std_smiles",
            "max_tanimoto_to_known_active",
            "novelty_bucket",
            "mw",
            "logp",
            "tpsa",
        ],
    )

    novelty_counts = {
        "high_novelty": sum(1 for r in analyzed if r["novelty_bucket"] == "high_novelty"),
        "moderate_novelty": sum(1 for r in analyzed if r["novelty_bucket"] == "moderate_novelty"),
        "low_novelty": sum(1 for r in analyzed if r["novelty_bucket"] == "low_novelty"),
    }

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Top-Hit Interpretation ({args.target.upper()})",
        "",
        "## Scope",
        f"- Ranked input: `{args.ranked_csv}`",
        f"- Known actives: `{args.known_actives_csv}`",
        f"- Top-N analyzed: {len(analyzed)}",
        "",
        "## Novelty snapshot",
        f"- High novelty (<0.30 Tanimoto): {novelty_counts['high_novelty']}",
        f"- Moderate novelty (0.30-0.60): {novelty_counts['moderate_novelty']}",
        f"- Low novelty (>=0.60): {novelty_counts['low_novelty']}",
        "",
        "## Interpretation template",
        "- Primary chemotype(s): TODO",
        "- Recurrent substructure motifs: TODO",
        "- Potential liabilities (reactivity/solubility/size): TODO",
        "- Suggested follow-up (rescore, redock, analog expansion): TODO",
        "",
        "## Top hits (first 5)",
    ]
    for row in analyzed[:5]:
        lines.append(
            f"- Rank {row['rank']} `{row['compound_id']}` | score={row['predicted_score_kcal_mol']} | "
            f"Tanimoto={row['max_tanimoto_to_known_active']} | novelty={row['novelty_bucket']}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv.as_posix()} and {out_md.as_posix()}")


if __name__ == "__main__":
    main()
