"""
Build a leakage-controlled train/val/test split.

The default seed-42 random split in data/splits.json lets near-duplicate ligands land in both
train and test, which inflates apparent generalization (the PDBbind time-split / CleanSplit /
Leak-Proof PDBBind literature). This script clusters complexes by ligand similarity and assigns
*whole clusters* to a single split, so no test ligand has a near-duplicate in train.

Method (Butina clustering on Morgan fingerprints):
  1. Read the processed .pt files, recover each ligand's canonical SMILES.
  2. Morgan fingerprint (radius 2, 2048 bits) per ligand.
  3. Butina cluster at Tanimoto distance cutoff (default 0.4 => similarity >= 0.6 groups).
  4. Greedily assign clusters to test/val/train until target fractions are met, largest
     clusters first into train. Every member of a cluster goes to the same split.
  5. Write splits_clean.json with the same schema as data/splits.json.

A leakage report (max train-vs-test Tanimoto) is printed and saved.

Example:
  python scripts/build_clean_split.py --processed_dir data/processed \
      --out data/splits_clean.json --cutoff 0.4 --val_frac 0.1 --test_frac 0.1
"""

import argparse
import glob
import json
import os
import sys

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_smiles(processed_dir: str):
    """Return (ids, smiles_list) for every .pt file that yields a parseable ligand."""
    ids, smiles = [], []
    for path in sorted(glob.glob(os.path.join(processed_dir, "*.pt"))):
        cid = os.path.splitext(os.path.basename(path))[0]
        try:
            data = torch.load(path, weights_only=False)
        except Exception:
            continue
        smi = getattr(data, "ligand_meta_canonical_smiles", None) or getattr(data, "smiles", None)
        if smi is None:
            continue
        ids.append(cid)
        smiles.append(smi)
    return ids, smiles


def _fingerprints(smiles):
    fps = []
    keep = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        keep.append(i)
    return fps, keep


def _butina_clusters(fps, cutoff: float):
    """Return list of clusters; each cluster is a tuple of fingerprint indices."""
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def _cross_tanimoto(fps, train_idx, test_idx, sample_cap=2000):
    """Leakage stats: for each test ligand take its max Tanimoto to any train ligand, then
    report the overall max and the median of those per-test maxima (the better indicator)."""
    import numpy as np
    train_fps = [fps[j] for j in train_idx]
    per_test_max = []
    for ti in test_idx[:sample_cap]:
        sims = DataStructs.BulkTanimotoSimilarity(fps[ti], train_fps)
        if sims:
            per_test_max.append(max(sims))
    if not per_test_max:
        return {"max": None, "median_per_test_max": None}
    return {
        "max": round(float(max(per_test_max)), 3),
        "median_per_test_max": round(float(np.median(per_test_max)), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--out", default="data/splits_clean.json")
    p.add_argument("--cutoff", type=float, default=0.4,
                   help="Butina Tanimoto DISTANCE cutoff (0.4 => groups ligands with sim >= 0.6).")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--report", default=None, help="Optional path for the leakage report JSON.")
    args = p.parse_args()

    ids, smiles = _load_smiles(args.processed_dir)
    print(f"Loaded {len(ids)} complexes with SMILES.")
    fps, keep = _fingerprints(smiles)
    kept_ids = [ids[i] for i in keep]
    print(f"Fingerprinted {len(fps)} ligands.")

    clusters = _butina_clusters(fps, args.cutoff)
    # Sort clusters largest-first for stable, deterministic assignment.
    clusters = sorted(clusters, key=len, reverse=True)
    print(f"Formed {len(clusters)} clusters (largest={len(clusters[0]) if clusters else 0}).")

    n = len(fps)
    n_test_target = int(round(args.test_frac * n))
    n_val_target = int(round(args.val_frac * n))

    # Assign smallest clusters to test/val first (keeps big well-sampled families in train),
    # then everything else to train. Whole clusters never split across sets.
    test_set, val_set, train_set = [], [], []
    for cl in sorted(clusters, key=len):  # small clusters first
        members = list(cl)
        if len(test_set) < n_test_target:
            test_set.extend(members)
        elif len(val_set) < n_val_target:
            val_set.extend(members)
        else:
            train_set.extend(members)

    split_idx = {"train": train_set, "val": val_set, "test": test_set}
    splits = {k: sorted(kept_ids[i] for i in v) for k, v in split_idx.items()}

    leak = _cross_tanimoto(fps, train_set, test_set)
    report = {
        "n_total": n,
        "n_clusters": len(clusters),
        "cutoff": args.cutoff,
        "sizes": {k: len(v) for k, v in splits.items()},
        "max_train_test_tanimoto": leak["max"],
        "median_per_test_max_tanimoto": leak["median_per_test_max"],
        "note": ("Compare against the random split (where identical ligands, Tanimoto=1.0, "
                 "appear in both train and test). Lower median_per_test_max => less leakage; "
                 "tighten --cutoff to reduce it further."),
    }

    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    report_path = args.report or (os.path.splitext(args.out)[0] + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote split -> {args.out}\nWrote report -> {report_path}")


if __name__ == "__main__":
    main()
