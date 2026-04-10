"""
Training loop for SE(3) flow matching ligand conformation model.

Usage:
    python -m src.training.train \
        --processed_dir data/processed \
        --splits        data/splits.json \
        --hidden_dim    64 \
        --n_layers      4 \
        --batch_size    8 \
        --lr            3e-4 \
        --n_epochs      100 \
        --patience      15 \
        --checkpoint_dir checkpoints \
        --wandb_project flow-match \
        --device        auto
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.models.egnn import EGNNFlowModel, count_parameters, build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import compute_test_metrics, kabsch_rmsd


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(preference: str = "auto") -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: EGNNFlowModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_rmsd: float,
):
    torch.save({
        "epoch":           epoch,
        "val_rmsd":        val_rmsd,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(
    path: str,
    model: EGNNFlowModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_epochs: int = 5,
):
    """Linear warmup → CosineAnnealingLR."""
    warmup = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    flow_matcher: FlowMatcher,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """One training epoch. Returns mean loss."""
    flow_matcher.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        loss = flow_matcher.compute_loss(batch)
        loss.backward()

        nn.utils.clip_grad_norm_(flow_matcher.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(
    flow_matcher: FlowMatcher,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validation epoch. Returns (mean_val_loss, median_rmsd)."""
    flow_matcher.eval()
    total_loss = 0.0
    n_batches = 0
    rmsd_list = []

    for batch in loader:
        batch = batch.to(device)
        loss = flow_matcher.compute_loss(batch)
        total_loss += loss.item()
        n_batches += 1

        # Generate one sample per molecule and compute RMSD
        generated = flow_matcher.generate(batch, n_steps=flow_matcher.n_steps)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = lig_batch.max().item() + 1

        for g in range(n_graphs):
            mask = lig_batch == g
            crystal_g = crystal_pos[mask].cpu()
            gen_g     = generated[g].cpu()
            rmsd_list.append(kabsch_rmsd(gen_g, crystal_g))

    import numpy as np
    val_loss = total_loss / max(n_batches, 1)
    median_rmsd = float(np.median(rmsd_list)) if rmsd_list else float("inf")
    return val_loss, median_rmsd


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: argparse.Namespace):
    device = detect_device(config.device)
    print(f"Using device: {device}")

    # --- Data ---
    splits = load_splits(config.splits)
    train_ds = PDBBindDataset(config.processed_dir, splits["train"])
    val_ds   = PDBBindDataset(config.processed_dir, splits["val"])
    test_ds  = PDBBindDataset(config.processed_dir, splits["test"])

    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=config.batch_size, shuffle=False)
    test_loader  = make_dataloader(test_ds,  batch_size=config.batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    model = build_default_model(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    assert n_params < 500_000, (
        f"Model has {n_params:,} params — exceeds 500K limit. "
        "Reduce hidden_dim or n_layers."
    )

    flow_matcher = FlowMatcher(model, n_steps=config.n_inference_steps).to(device)

    # --- Optimizer and scheduler ---
    optimizer = AdamW(flow_matcher.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, config.n_epochs, warmup_epochs=5)

    # --- Wandb ---
    if WANDB_AVAILABLE and config.wandb_project:
        wandb.init(project=config.wandb_project, config=vars(config))

    # --- Training loop ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_rmsd = float("inf")
    patience_count = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    for epoch in range(1, config.n_epochs + 1):
        train_loss = train_epoch(flow_matcher, train_loader, optimizer, device)
        val_loss, val_rmsd = val_epoch(flow_matcher, val_loader, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{config.n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmsd={val_rmsd:.3f} Å | "
            f"lr={current_lr:.2e}"
        )

        if WANDB_AVAILABLE and config.wandb_project:
            wandb.log({
                "epoch":       epoch,
                "train/loss":  train_loss,
                "val/loss":    val_loss,
                "val/rmsd_median": val_rmsd,
                "lr":          current_lr,
            })

        # Early stopping on val RMSD
        if val_rmsd < best_val_rmsd:
            best_val_rmsd = val_rmsd
            patience_count = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, val_rmsd)
            print(f"  ✓ New best val RMSD: {val_rmsd:.3f} Å (saved checkpoint)")
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

    print(f"\nBest val RMSD: {best_val_rmsd:.3f} Å")

    # --- Test evaluation ---
    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model)
    flow_matcher.eval()

    test_metrics = compute_test_metrics(flow_matcher, test_loader, device)
    print("\n=== Test Metrics ===")
    print(f"RMSD median:   {test_metrics['rmsd_median']:.3f} Å")
    print(f"RMSD mean:     {test_metrics['rmsd_mean']:.3f} Å")
    print(f"RMSD < 1 Å:    {test_metrics['rmsd_pct_under_1A']:.1f}%")
    print(f"RMSD < 2 Å:    {test_metrics['rmsd_pct_under_2A']:.1f}%")
    print(f"RMSD < 5 Å:    {test_metrics['rmsd_pct_under_5A']:.1f}%")
    print(f"Strain median: {test_metrics['strain_median']:.3f}")
    print(f"MMFF failures: {test_metrics['n_mmff_failed']} / {test_metrics['n_total']}")

    if WANDB_AVAILABLE and config.wandb_project:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SE(3) flow matching model")
    p.add_argument("--processed_dir",    default="data/processed")
    p.add_argument("--splits",           default="data/splits.json")
    p.add_argument("--hidden_dim",       type=int,   default=64)
    p.add_argument("--n_layers",         type=int,   default=4)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--n_epochs",         type=int,   default=100)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--n_inference_steps", type=int,  default=20)
    p.add_argument("--checkpoint_dir",   default="checkpoints")
    p.add_argument("--wandb_project",    default="")
    p.add_argument("--device",           default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
