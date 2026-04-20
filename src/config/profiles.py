"""
Hardware profiles for flow matching training.

Each profile tunes: model size, batch size, AMP dtype, DataLoader workers,
torch.compile, and inference steps for a specific accelerator.

Usage:
    python -m src.training.train --profile l4
    python -m src.training.train --profile a100 --batch_size 64  # override one param
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    # Model architecture
    hidden_dim: int
    n_layers: int
    max_params: int
    # DataLoader
    batch_size: int
    num_workers: int
    # Optimiser
    lr: float
    # Early stopping — higher on GPU because val RMSD is noisier at fewer steps
    patience: int
    # AMP: None = disabled, 'float16' (T4/L4/G4), 'bfloat16' (A100/H100)
    amp_dtype: Optional[str]
    # torch.compile (PyTorch 2.x) — skip on T4/G4, dynamic shapes matter
    compile_model: bool
    # Euler ODE steps
    n_inference_steps: int
    val_inference_steps: int
    # Default device string passed to detect_device()
    default_device: str


PROFILES: dict[str, HardwareProfile] = {
    # ------------------------------------------------------------------ CPU
    "cpu": HardwareProfile(
        name="cpu",
        hidden_dim=64,      n_layers=4,   max_params=500_000,
        batch_size=8,       num_workers=0,
        lr=3e-4,            patience=25,
        amp_dtype=None,     compile_model=False,
        n_inference_steps=20, val_inference_steps=5,
        default_device="cpu",
    ),
    # ------------------------------------------------------------------ T4 (15 GB VRAM, 2 vCPU)
    "t4": HardwareProfile(
        name="t4",
        hidden_dim=128,     n_layers=4,   max_params=2_000_000,
        batch_size=32,      num_workers=2,
        lr=3e-4,            patience=25,
        amp_dtype="float16", compile_model=False,
        n_inference_steps=50, val_inference_steps=15,
        default_device="cuda",
    ),
    # ------------------------------------------------------------------ G4 (similar to T4 in Colab)
    "g4": HardwareProfile(
        name="g4",
        hidden_dim=128,     n_layers=4,   max_params=2_000_000,
        batch_size=32,      num_workers=2,
        lr=3e-4,            patience=25,
        amp_dtype="float16", compile_model=False,
        n_inference_steps=50, val_inference_steps=15,
        default_device="cuda",
    ),
    # ------------------------------------------------------------------ L4 (24 GB VRAM, 4 vCPU)
    "l4": HardwareProfile(
        name="l4",
        hidden_dim=128,     n_layers=6,   max_params=3_000_000,
        batch_size=64,      num_workers=4,
        lr=3e-4,            patience=25,
        amp_dtype="float16", compile_model=True,
        n_inference_steps=50, val_inference_steps=15,
        default_device="cuda",
    ),
    # ------------------------------------------------------------------ A100 (40/80 GB VRAM, 4 vCPU)
    "a100": HardwareProfile(
        name="a100",
        hidden_dim=256,     n_layers=6,   max_params=10_000_000,
        batch_size=128,     num_workers=4,
        lr=3e-4,            patience=30,
        amp_dtype="bfloat16", compile_model=True,
        n_inference_steps=100, val_inference_steps=20,
        default_device="cuda",
    ),
    # ------------------------------------------------------------------ H100 (80 GB VRAM, 4 vCPU)
    "h100": HardwareProfile(
        name="h100",
        hidden_dim=256,     n_layers=8,   max_params=10_000_000,
        batch_size=256,     num_workers=4,
        lr=3e-4,            patience=30,
        amp_dtype="bfloat16", compile_model=True,
        n_inference_steps=100, val_inference_steps=20,
        default_device="cuda",
    ),
}
