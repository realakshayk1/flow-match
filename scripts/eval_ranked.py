"""
Multi-pose + confidence-ranked evaluation.

For each complex: sample N poses, score each with the confidence head, and report:
  - top-1 (confidence-ranked): success of the single highest-confidence pose
  - top-1 (random):            success of an arbitrary pose (pose 0) — the no-ranking baseline
  - top-K (oracle):            success if ANY of the N poses is correct (upper bound)
  - selective-prediction curve: success rate vs confidence percentile (the DiffDock 38->83 lift)

Success = in-frame symmetry-corrected dock RMSD < 2 A.

Example:
  python scripts/eval_ranked.py \
      --checkpoint checkpoints/best_model_conf.pt \
      --split test --n_poses 10 --n_steps 20 --out_dir results/phase2/ranked
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.models.egnn import build_default_model, load_model_state
from src.models.flow_model import FlowMatcher
from src.training.metrics import symmetry_rmsd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Checkpoint WITH a trained confidence head")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_poses", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--success_rmsd", type=float, default=2.0)
    p.add_argument("--max_complexes", type=int, default=None)
    p.add_argument("--out_dir", default="results/phase2/ranked")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("has_confidence"):
        print("WARNING: checkpoint has no 'has_confidence' marker; confidence head may be untrained.")
    rc = ckpt.get("run_config", {})
    hidden_dim = rc.get("hidden_dim", ckpt["model_state"]["lig_emb.weight"].shape[0])
    layer_idx = [int(k.split(".")[1]) for k in ckpt["model_state"] if k.startswith("layers.")]
    n_layers = rc.get("n_layers", max(layer_idx) + 1 if layer_idx else 4)

    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers, with_confidence=True)
    load_model_state(model, ckpt["model_state"])
    model = model.to(device)
    flow_matcher = FlowMatcher(model, n_steps=args.n_steps).to(device)
    flow_matcher.eval()

    splits = load_splits(args.splits)
    ids = splits[args.split]
    if args.max_complexes:
        ids = ids[: args.max_complexes]
    ds = PDBBindDataset(args.processed_dir, ids)
    loader = make_dataloader(ds, batch_size=args.batch_size, shuffle=False)

    # Per-complex records: confidence of chosen pose + success flags.
    records = []  # {cid, conf_top, ranked_ok, random_ok, oracle_ok}
    for batch in loader:
        batch = batch.to(device)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = int(lig_batch.max().item()) + 1
        smiles_list = getattr(batch, "ligand_meta_canonical_smiles", None)
        cids = batch.complex_id if isinstance(batch.complex_id, list) else [batch.complex_id]

        poses = flow_matcher.generate_multi(batch, n_poses=args.n_poses, n_steps=args.n_steps)
        # confidence per pose: [n_poses][n_graphs]
        confs = [flow_matcher.confidence_scores(batch, pose).cpu().numpy() for pose in poses]

        for g in range(n_graphs):
            if smiles_list is None or g >= len(smiles_list):
                continue
            mol = Chem.MolFromSmiles(smiles_list[g])
            if mol is None:
                continue
            crys = crystal_pos[lig_batch == g].cpu().numpy()
            rmsds, gconfs = [], []
            for pi in range(len(poses)):
                gen = poses[pi][g].cpu().numpy()
                if mol.GetNumAtoms() != gen.shape[0]:
                    rmsds.append(float("nan"))
                else:
                    rmsds.append(symmetry_rmsd(mol, gen, crys))
                gconfs.append(float(confs[pi][g]))
            rmsds = np.array(rmsds)
            gconfs = np.array(gconfs)
            if np.all(np.isnan(rmsds)):
                continue

            ranked_idx = int(np.nanargmax(gconfs))
            oracle_ok = bool(np.nanmin(rmsds) < args.success_rmsd)
            records.append({
                "complex_id": cids[g],
                "conf_top": float(gconfs[ranked_idx]),
                "ranked_ok": bool(rmsds[ranked_idx] < args.success_rmsd),
                "random_ok": bool(rmsds[0] < args.success_rmsd),
                "oracle_ok": oracle_ok,
                "ranked_rmsd": float(rmsds[ranked_idx]),
            })

    n = len(records)
    if n == 0:
        print("No scorable complexes.")
        return

    ranked = np.array([r["ranked_ok"] for r in records], dtype=float)
    random_ = np.array([r["random_ok"] for r in records], dtype=float)
    oracle = np.array([r["oracle_ok"] for r in records], dtype=float)
    conf = np.array([r["conf_top"] for r in records], dtype=float)

    # Selective-prediction curve: keep the most-confident fraction, report success there.
    order = np.argsort(-conf)
    curve = []
    for frac in (1.0, 0.66, 0.5, 0.33, 0.1):
        k = max(1, int(round(frac * n)))
        keep = order[:k]
        curve.append({"keep_frac": frac, "n": k,
                      "ranked_success_pct": round(float(ranked[keep].mean() * 100), 1)})

    summary = {
        "n_total": n,
        "n_poses": args.n_poses,
        "success_rmsd": args.success_rmsd,
        "top1_random_success_pct": round(float(random_.mean() * 100), 1),
        "top1_ranked_success_pct": round(float(ranked.mean() * 100), 1),
        "topk_oracle_success_pct": round(float(oracle.mean() * 100), 1),
        "selective_curve": curve,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ranked_summary.json"), "w") as f:
        json.dump({"summary": summary, "per_complex": records}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {os.path.join(args.out_dir, 'ranked_summary.json')}")


if __name__ == "__main__":
    main()
