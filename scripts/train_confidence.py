"""
Train a confidence head on top of a trained Flow-Match base model.

Recipe (DiffDock-style, shared trunk):
  1. Load a trained base checkpoint and rebuild the model WITH a confidence head
     (base weights loaded; head initialised randomly).
  2. For each training complex, sample N poses, label each pose by
     (in-frame symmetry-corrected dock RMSD < threshold).
  3. Train the confidence head (BCE) to predict that label. The trunk is frozen by default
     so the velocity field is untouched; pass --finetune_trunk to update it too.
  4. Save a checkpoint whose model_state includes the confidence head and a
     "has_confidence": True marker.

At inference, score each of N sampled poses with FlowMatcher.confidence_scores and report the
argmax as the confidence-ranked top-1.

Example:
  python scripts/train_confidence.py \
      --base_checkpoint checkpoints/best_model.pt \
      --out_checkpoint checkpoints/best_model_conf.pt \
      --split train --n_poses 8 --n_steps 20 --epochs 5 --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.models.egnn import build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import symmetry_rmsd


def _pose_labels(batch, pose, threshold: float) -> torch.Tensor:
    """Label each graph's pose: 1.0 if dock RMSD < threshold, else 0.0. NaN RMSD -> 0.0."""
    lig_batch = batch["ligand"].batch
    crystal_pos = batch["ligand"].pos
    n_graphs = int(lig_batch.max().item()) + 1
    smiles_list = getattr(batch, "ligand_meta_canonical_smiles", None)
    labels = torch.zeros(n_graphs)
    for g in range(n_graphs):
        if smiles_list is None or g >= len(smiles_list):
            continue
        mol = Chem.MolFromSmiles(smiles_list[g])
        gen = pose[g].cpu().numpy()
        if mol is None or mol.GetNumAtoms() != gen.shape[0]:
            continue
        crys = crystal_pos[lig_batch == g].cpu().numpy()
        rmsd = symmetry_rmsd(mol, gen, crys)
        if not np.isnan(rmsd) and rmsd < threshold:
            labels[g] = 1.0
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--out_checkpoint", required=True)
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n_poses", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=2.0, help="dock RMSD (A) success cutoff")
    p.add_argument("--finetune_trunk", action="store_true",
                   help="Also update the EGNN trunk (default: train confidence head only).")
    p.add_argument("--max_complexes", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    rc = ckpt.get("run_config", {})
    hidden_dim = rc.get("hidden_dim", ckpt["model_state"]["lig_emb.weight"].shape[0])
    layer_idx = [int(k.split(".")[1]) for k in ckpt["model_state"] if k.startswith("layers.")]
    n_layers = rc.get("n_layers", max(layer_idx) + 1 if layer_idx else 4)

    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers, with_confidence=True)
    # Load base weights; confidence head stays at its random init.
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded base. Missing (expected = confidence head): {missing}")
    model = model.to(device)
    flow_matcher = FlowMatcher(model, n_steps=args.n_steps).to(device)

    if not args.finetune_trunk:
        for name, prm in model.named_parameters():
            if "confidence_head" not in name:
                prm.requires_grad_(False)
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)}")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    splits = load_splits(args.splits)
    ids = splits[args.split]
    if args.max_complexes:
        ids = ids[: args.max_complexes]
    ds = PDBBindDataset(args.processed_dir, ids)
    loader = make_dataloader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"Training confidence head on {len(ds)} {args.split} complexes, "
          f"{args.n_poses} poses each.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n_batches, pos_frac = 0.0, 0, []
        for batch in loader:
            batch = batch.to(device)
            poses = flow_matcher.generate_multi(batch, n_poses=args.n_poses, n_steps=args.n_steps)
            batch_loss = 0.0
            for pose in poses:
                labels = _pose_labels(batch, pose, args.threshold).to(device)
                pos_frac.append(labels.mean().item())
                batch_loss = batch_loss + flow_matcher.confidence_loss(batch, pose, labels)
            batch_loss = batch_loss / len(poses)
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
            total += batch_loss.item()
            n_batches += 1
        print(f"Epoch {epoch}/{args.epochs} | conf_bce={total / max(n_batches,1):.4f} | "
              f"pos_label_frac={np.mean(pos_frac) if pos_frac else 0.0:.3f}")

    out = {
        "epoch": ckpt.get("epoch"),
        "val_rmsd": ckpt.get("val_rmsd"),
        "model_state": model.state_dict(),
        "run_config": {**rc, "hidden_dim": hidden_dim, "n_layers": n_layers},
        "has_confidence": True,
        "confidence_threshold": args.threshold,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_checkpoint)), exist_ok=True)
    torch.save(out, args.out_checkpoint)
    print(f"Saved confidence checkpoint to {args.out_checkpoint}")


if __name__ == "__main__":
    main()
