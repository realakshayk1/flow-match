"""
Training loop for SE(3) flow matching ligand conformation model.

Usage:
    # Use a hardware profile (sets model size, batch, AMP, workers, etc.)
    python -m src.training.train --profile l4
    python -m src.training.train --profile a100
    python -m src.training.train --profile cpu

    # Override individual profile params
    python -m src.training.train --profile l4 --batch_size 32

    # Full manual control (no profile)
    python -m src.training.train \\
        --processed_dir data/processed \\
        --splits        data/splits.json \\
        --hidden_dim    64 \\
        --n_layers      4 \\
        --batch_size    8 \\
        --lr            3e-4 \\
        --n_epochs      100 \\
        --patience      15 \\
        --checkpoint_dir checkpoints \\
        --wandb_project flow-match \\
        --device        auto
"""

import argparse
import os
import platform
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

from rdkit import Chem as _Chem

from src.config.profiles import PROFILES, HardwareProfile
from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.models.egnn import EGNNFlowModel, count_parameters, build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import compute_test_metrics, compute_etkdg_baseline, kabsch_rmsd


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(preference: str = "auto") -> torch.device:
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
    config: Optional[argparse.Namespace] = None,
):
    ckpt = {
        "epoch":           epoch,
        "val_rmsd":        val_rmsd,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if config is not None:
        ckpt["run_config"] = {
            "profile":    config.profile,
            "hidden_dim": config.hidden_dim,
            "n_layers":   config.n_layers,
            "batch_size": config.batch_size,
            "lr":         config.lr,
        }
    torch.save(ckpt, path)


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
# AMP helpers
# ---------------------------------------------------------------------------

def _build_amp(profile: HardwareProfile, device: torch.device):
    """Return (use_amp, amp_torch_dtype, scaler)."""
    use_amp = profile.amp_dtype is not None and device.type == "cuda"
    amp_torch_dtype = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(profile.amp_dtype or "", torch.float32)

    # GradScaler only needed for float16; bfloat16 has sufficient dynamic range.
    use_scaler = use_amp and amp_torch_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    return use_amp, amp_torch_dtype, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_full_test_evaluation(
    flow_matcher,
    model,
    test_loader,
    test_ds,
    device,
    config,
    best_ckpt_path
):
    """
    Unified end-of-training and eval-only testing execution block.
    """
    if os.path.exists(best_ckpt_path):
        print(f"\n--- Checkpoint Load Verification ---")
        print(f"Checkpoint path: {best_ckpt_path}")
        print(f"Checkpoint exists: True")
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        if "epoch" in ckpt:
            print(f"Epoch/Step: {ckpt['epoch']}")
        
        pre_sum = sum(p.abs().sum().item() for p in model.parameters())
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        post_sum = sum(p.abs().sum().item() for p in model.parameters())
        
        print(f"Missing keys: {len(missing)} ({missing[:3]}{'...' if len(missing)>3 else ''})")
        print(f"Unexpected keys: {len(unexpected)} ({unexpected[:3]}{'...' if len(unexpected)>3 else ''})")
        print(f"Total param checksum before: {pre_sum:.4f}")
        print(f"Total param checksum after:  {post_sum:.4f}")
        print(f"------------------------------------\n")
    flow_matcher.eval()

    test_metrics = compute_test_metrics(
        flow_matcher, 
        test_loader, 
        device,
        debug_eval_examples=getattr(config, 'debug_eval_examples', 0),
        dump_eval_predictions=getattr(config, 'dump_eval_predictions', '')
    )
    print("\n=== Test Metrics (Resume-safe) ===")
    print(f"RMSD median:   {test_metrics['rmsd_median']:.3f} Å")
    print(f"RMSD mean:     {test_metrics['rmsd_mean']:.3f} Å")
    print(f"RMSD < 1 Å:    {test_metrics['rmsd_pct_under_1A']:.1f}%")
    print(f"RMSD < 2 Å:    {test_metrics['rmsd_pct_under_2A']:.1f}%")
    print(f"RMSD < 5 Å:    {test_metrics['rmsd_pct_under_5A']:.1f}%")
    print("\n=== Chemistry Metrics (Experimental) ===")
    print(f"Strain median:         {test_metrics['strain_median']:.3f}")
    failures = test_metrics.get('strain_failures', {})
    print(f"Invalid geometry:      {failures.get('invalid_pose_geometry', 0)}")
    print(f"RDKit/MMFF setup fail: {failures.get('rdkit_mmff_setup_failure', 0)}")
    print(f"Absurd energy (>10k):  {failures.get('absurd_energy', 0)}")
    print(f"Other failures:        {failures.get('other', 0)}")

    if WANDB_AVAILABLE and config.wandb_project:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})

    # --- ETKDG baseline (requires SMILES stored in .pt files by preprocess.py) ---
    mol_lookup = {}
    for data in test_ds:
        if hasattr(data, "smiles") and data.smiles:
            mol = _Chem.MolFromSmiles(data.smiles)
            if mol is not None:
                mol_lookup[data.complex_id] = mol

    if mol_lookup:
        print(f"\nComputing ETKDG baseline on {len(mol_lookup)} molecules ...")
        etkdg_metrics = compute_etkdg_baseline(test_loader, mol_lookup)
        delta = test_metrics["rmsd_median"] - etkdg_metrics["rmsd_median"]
        print("\n=== ETKDG Baseline (Experimental) ===")
        print(f"RMSD median:   {etkdg_metrics['rmsd_median']:.3f} Å")
        print(f"RMSD mean:     {etkdg_metrics['rmsd_mean']:.3f} Å")
        print(f"RMSD < 1 Å:    {etkdg_metrics['rmsd_pct_under_1A']:.1f}%")
        print(f"RMSD < 2 Å:    {etkdg_metrics['rmsd_pct_under_2A']:.1f}%")
        print(f"RMSD < 5 Å:    {etkdg_metrics['rmsd_pct_under_5A']:.1f}%")
        print(f"Failures breakdown: {etkdg_metrics.get('failures', {})}")
        
        if delta < 0:
            print(f"Model is {abs(delta):.3f} Å better median RMSD than ETKDG baseline")
        else:
            print(f"Model is {delta:.3f} Å worse median RMSD than ETKDG baseline")

        if WANDB_AVAILABLE and config.wandb_project:
            wandb.log({f"etkdg/{k}": v for k, v in etkdg_metrics.items()
                       if isinstance(v, (int, float))})
            wandb.log({"etkdg/improvement_vs_model": -delta})
    else:
        print("\nSkipping ETKDG baseline — no SMILES found in test set.")
        print("Re-run preprocess.py to add SMILES to .pt files.")

    if WANDB_AVAILABLE and config.wandb_project:
        wandb.finish()


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    flow_matcher: FlowMatcher,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    flow_matcher.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matcher.compute_loss(batch)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(flow_matcher.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(flow_matcher.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.detach()
        n_batches += 1

    return (total_loss / max(n_batches, 1)).item()


@torch.no_grad()
def val_epoch(
    flow_matcher: FlowMatcher,
    loader,
    device: torch.device,
    inference_steps: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> Tuple[float, float]:
    flow_matcher.eval()
    total_loss = 0.0
    n_batches = 0
    rmsd_list = []

    n_steps = inference_steps if inference_steps is not None else flow_matcher.n_steps

    for batch in loader:
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matcher.compute_loss(batch)
        total_loss += loss.detach()
        n_batches += 1

        generated = flow_matcher.generate(batch, n_steps=n_steps)
        lig_batch = batch["ligand"].batch
        crystal_pos = batch["ligand"].pos
        n_graphs = int(lig_batch.max().item()) + 1

        for g in range(n_graphs):
            mask = lig_batch == g
            crystal_g = crystal_pos[mask].cpu()
            gen_g     = generated[g].cpu()
            rmsd_list.append(kabsch_rmsd(gen_g, crystal_g))

    import numpy as np
    val_loss = (total_loss / max(n_batches, 1)).item()
    median_rmsd = float(np.median(rmsd_list)) if rmsd_list else float("inf")
    return val_loss, median_rmsd


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: argparse.Namespace):
    from dataclasses import replace
    profile: HardwareProfile = PROFILES[config.profile]

    # --inspect_checkpoint: dump checkpoint metadata and exit without running eval
    if getattr(config, "inspect_checkpoint", False):
        ckpt_path = config.resume_checkpoint if config.resume_checkpoint else os.path.join(config.checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        rc = ckpt.get("run_config", {})
        print(f"=== Checkpoint Inspection: {ckpt_path} ===")
        print(f"  Checkpoint keys : {list(ckpt.keys())}")
        print(f"  Epoch           : {ckpt.get('epoch', 'n/a')}")
        print(f"  Val RMSD        : {ckpt.get('val_rmsd', 'n/a')}")
        print(f"  run_config      : {rc if rc else '(not saved)'}")
        if rc:
            hd = rc.get("hidden_dim", "?")
            nl = rc.get("n_layers", "?")
            print(f"  Trained profile : {rc.get('profile', '?')}")
            print(f"  hidden_dim      : {hd}")
            print(f"  n_layers        : {nl}")
            if isinstance(hd, int) and isinstance(nl, int):
                m = build_default_model(hidden_dim=hd, n_layers=nl)
                print(f"  Param count     : {count_parameters(m):,}")
        keys = list(ckpt["model_state"].keys())
        print(f"  State dict keys : {len(keys)} ({keys[0]} ... {keys[-1]})")
        print("==========================================")
        return

    if hasattr(config, 'compile_model_override') and config.compile_model_override is not None:
        profile = replace(profile, compile_model=(config.compile_model_override == "True"))

    if hasattr(config, 'amp_dtype_override') and config.amp_dtype_override is not None:
        val = None if config.amp_dtype_override == "disabled" else config.amp_dtype_override
        profile = replace(profile, amp_dtype=val)


    device = detect_device(config.device)
    print(f"Profile : {profile.name}")
    print(f"Device  : {device}")
    print(f"AMP     : {profile.amp_dtype or 'disabled'}")
    print(f"Compile : {profile.compile_model}")
    print(f"Workers : {config.num_workers}")
    print()

    # --- Data ---
    splits = load_splits(config.splits)
    train_ds = PDBBindDataset(config.processed_dir, splits["train"])
    val_ds   = PDBBindDataset(config.processed_dir, splits["val"])
    test_ds  = PDBBindDataset(config.processed_dir, splits["test"])
    if getattr(config, "max_test_examples", 0) > 0:
        test_ds.complex_ids = test_ds.complex_ids[:config.max_test_examples]
        if test_ds._cache is not None:
            test_ds._cache = test_ds._cache[:config.max_test_examples]


    pin = device.type == "cuda"
    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True,
                                   num_workers=config.num_workers, pin_memory=pin)
    val_loader   = make_dataloader(val_ds,   batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=pin)
    test_loader  = make_dataloader(test_ds,  batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=pin)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    if config.eval_only:
        best_ckpt_path = config.resume_checkpoint if config.resume_checkpoint else os.path.join(config.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            run_config = ckpt.get("run_config", {})
            print(f"--- Checkpoint Architecture Probe ---")
            print(f"  run_config in checkpoint: {bool(run_config)}")
            if run_config:
                print(f"  checkpoint trained with profile : {run_config.get('profile', 'unknown')}")
                print(f"  checkpoint hidden_dim           : {run_config.get('hidden_dim', 'not saved')}")
                print(f"  checkpoint n_layers             : {run_config.get('n_layers', 'not saved')}")
                config.hidden_dim = run_config.get("hidden_dim", config.hidden_dim)
                config.n_layers = run_config.get("n_layers", config.n_layers)
            else:
                print(f"  No run_config — using profile defaults: hidden_dim={config.hidden_dim}, n_layers={config.n_layers}")
            print(f"  Effective hidden_dim : {config.hidden_dim}")
            print(f"  Effective n_layers   : {config.n_layers}")
            print(f"-------------------------------------")
        else:
            print(f"")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"  WARNING: --eval_only set but checkpoint NOT FOUND at:")
            print(f"    {best_ckpt_path}")
            print(f"  Model will run with RANDOM (untrained) weights.")
            print(f"  All RMSD results will be meaningless.")
            print(f"  Fix: mount Google Drive and pass --resume_checkpoint <path>")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"")

    model = build_default_model(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} (limit: {profile.max_params:,})")
    print(f"Architecture: hidden_dim={config.hidden_dim}, n_layers={config.n_layers}")
    if n_params > profile.max_params:
        raise ValueError(
            f"Model has {n_params:,} params — exceeds {profile.name} limit of "
            f"{profile.max_params:,}. Reduce --hidden_dim or --n_layers."
        )

    # torch.compile: fuse kernels for repeated graph shapes (PyTorch 2.x)
    if profile.compile_model and device.type == "cuda":
        print("Compiling model with torch.compile(dynamic=True) ...")
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it.")

    flow_matcher = FlowMatcher(model, n_steps=config.n_inference_steps).to(device)

    # --- AMP setup ---
    use_amp, amp_torch_dtype, scaler = _build_amp(profile, device)

    # --- Optimizer and scheduler ---
    optimizer = AdamW(flow_matcher.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, config.n_epochs, warmup_epochs=5)

    # --- Wandb ---
    if WANDB_AVAILABLE and config.wandb_project:
        wandb.init(project=config.wandb_project, config=vars(config))

    # --- Training loop ---
    best_ckpt_path = config.resume_checkpoint if config.resume_checkpoint else os.path.join(config.checkpoint_dir, "best_model.pt")

    if not config.eval_only:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        best_val_rmsd = float("inf")
        patience_count = 0

        for epoch in range(1, config.n_epochs + 1):
            train_loss = train_epoch(
                flow_matcher, train_loader, optimizer, device,
                use_amp=use_amp, amp_dtype=amp_torch_dtype, scaler=scaler,
            )
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            do_val = (epoch % config.val_freq == 0) or (epoch == config.n_epochs)
            if do_val:
                val_loss, val_rmsd = val_epoch(
                    flow_matcher, val_loader, device,
                    inference_steps=config.val_inference_steps,
                    use_amp=use_amp, amp_dtype=amp_torch_dtype,
                )
                print(
                    f"Epoch {epoch:3d}/{config.n_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_rmsd={val_rmsd:.3f} Å | "
                    f"lr={current_lr:.2e}"
                )
            else:
                val_loss, val_rmsd = float("nan"), float("nan")
                print(
                    f"Epoch {epoch:3d}/{config.n_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"(val skipped) | "
                    f"lr={current_lr:.2e}"
                )

            if WANDB_AVAILABLE and config.wandb_project:
                log_dict = {"epoch": epoch, "train/loss": train_loss, "lr": current_lr}
                if do_val:
                    log_dict["val/loss"] = val_loss
                    log_dict["val/rmsd_median"] = val_rmsd
                wandb.log(log_dict)

            if do_val and not (val_rmsd != val_rmsd):  # not NaN
                if val_rmsd < best_val_rmsd:
                    best_val_rmsd = val_rmsd
                    patience_count = 0
                    save_checkpoint(best_ckpt_path, model, optimizer, epoch, val_rmsd, config)
                    print(f"  ✓ New best val RMSD: {val_rmsd:.3f} Å (saved checkpoint)")
                else:
                    patience_count += 1
                    if patience_count >= config.patience:
                        print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                        break

        print(f"\nBest val RMSD: {best_val_rmsd:.3f} Å")

    # --- Test evaluation ---
    run_full_test_evaluation(
        flow_matcher=flow_matcher,
        model=model,
        test_loader=test_loader,
        test_ds=test_ds,
        device=device,
        config=config,
        best_ckpt_path=best_ckpt_path
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SE(3) flow matching model")

    p.add_argument("--profile", default="cpu", choices=list(PROFILES.keys()),
                   help="Hardware profile. Sets model size, batch size, AMP, workers, etc. "
                        "Individual args below override the profile when explicitly passed.")

    # Profile-tunable args: default=None means "use profile value"
    p.add_argument("--hidden_dim",          type=int,   default=None)
    p.add_argument("--n_layers",            type=int,   default=None)
    p.add_argument("--batch_size",          type=int,   default=None)
    p.add_argument("--lr",                  type=float, default=None)
    p.add_argument("--patience",            type=int,   default=None)
    p.add_argument("--n_inference_steps",   type=int,   default=None)
    p.add_argument("--val_inference_steps", type=int,   default=None)
    p.add_argument("--device",              type=str,   default=None,
                   help="Override device. Default: profile's default_device.")

    # Non-profile args
    p.add_argument("--processed_dir",   default="data/processed")
    p.add_argument("--splits",          default="data/splits.json")
    p.add_argument("--n_epochs",        type=int, default=100)
    p.add_argument("--val_freq",        type=int, default=1)
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--wandb_project",   default="")
    p.add_argument("--eval_only",       action="store_true")
    p.add_argument("--resume_checkpoint", default=None, type=str)
    p.add_argument("--inspect_checkpoint", action="store_true",
                   help="Print checkpoint metadata (architecture, epoch, val_rmsd) and exit. No eval run.")
    p.add_argument("--max_test_examples", type=int, default=0, help="If > 0, restrict test set to this many examples for debugging.")
    p.add_argument("--debug_eval_examples", type=int, default=0, help="Number of examples to dump verbose tensor outputs for.")
    p.add_argument("--dump_eval_predictions", type=str, default="", help="Path to write per-example RMSD and chemistry failure summary JSONL to.")

    p.add_argument("--compile_model_override", type=str, choices=["True", "False"], default=None,
                   help="Override compile_model profile setting. Use 'True' or 'False'.")
    p.add_argument("--amp_dtype_override", type=str, choices=["float16", "bfloat16", "disabled"], default=None,
                   help="Override AMP dtype profile setting.")

    args = p.parse_args()

    # Fill None args from profile
    profile = PROFILES[args.profile]
    if args.hidden_dim          is None: args.hidden_dim          = profile.hidden_dim
    if args.n_layers            is None: args.n_layers            = profile.n_layers
    if args.batch_size          is None: args.batch_size          = profile.batch_size
    if args.lr                  is None: args.lr                  = profile.lr
    if args.patience            is None: args.patience            = profile.patience
    if args.n_inference_steps   is None: args.n_inference_steps   = profile.n_inference_steps
    if args.val_inference_steps is None: args.val_inference_steps = profile.val_inference_steps
    if args.device              is None: args.device              = profile.default_device

    # num_workers: force 0 on Windows (no fork support)
    args.num_workers = profile.num_workers
    if platform.system() == "Windows" and args.num_workers > 0:
        print(f"Windows detected: overriding num_workers {args.num_workers} -> 0")
        args.num_workers = 0

    return args


if __name__ == "__main__":
    train(parse_args())
