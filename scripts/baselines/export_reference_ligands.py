"""
Export crystal reference ligand SDFs (one per complex) for a split, so that any docking method
— the Flow-Match model, Vina, GNINA, DiffDock — can be scored against a common reference with
``scripts/baselines/score_external_poses.py``.

Reads the raw PDBBind ligand (in the receptor coordinate frame) and writes ``{out_dir}/{cid}.sdf``.
This is the ``--ref_dir`` expected by the shared scorer.

Example:
  python scripts/baselines/export_reference_ligands.py \
      --splits data/splits_clean.json --split test \
      --raw_dir data/raw/refined-set --out_dir results/phase3/reference_ligands
"""

import argparse
import json
import os
import sys

from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_ligand(path):
    for sanitize in (True, False):
        supplier = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=True)
        for mol in supplier:
            if mol is not None:
                return mol
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits_clean.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--raw_dir", default="data/raw/refined-set")
    p.add_argument("--out_dir", default="results/phase3/reference_ligands")
    p.add_argument("--max_complexes", type=int, default=None)
    args = p.parse_args()

    with open(args.splits) as f:
        ids = json.load(f)[args.split]
    if args.max_complexes:
        ids = ids[: args.max_complexes]

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok, n_missing, n_bad = 0, 0, 0
    for cid in ids:
        lig_path = os.path.join(args.raw_dir, cid, f"{cid}_ligand.sdf")
        if not os.path.exists(lig_path):
            n_missing += 1
            continue
        mol = _load_ligand(lig_path)
        if mol is None:
            n_bad += 1
            continue
        w = Chem.SDWriter(os.path.join(args.out_dir, f"{cid}.sdf"))
        w.write(mol)
        w.close()
        n_ok += 1

    print(f"Wrote {n_ok} reference ligands to {args.out_dir} "
          f"(missing raw: {n_missing}, unparseable: {n_bad}) of {len(ids)} in split '{args.split}'")


if __name__ == "__main__":
    main()
