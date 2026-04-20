import argparse
import sys
import os

import torch
import numpy as np

# ensure src is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flow_model import FlowMatcher
from src.models.egnn import count_parameters, build_default_model
from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.training.metrics import compute_test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_inference_steps", type=int, default=100)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = ckpt.get("run_config", {})
    hidden_dim = run_config.get("hidden_dim", 64)
    n_layers = run_config.get("n_layers", 4)
    
    device = torch.device(args.device)

    splits = load_splits(args.splits)
    test_ds  = PDBBindDataset(args.processed_dir, splits["test"])
    test_loader  = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_ds)} test complexes.")

    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    flow_matcher = FlowMatcher(model, n_steps=args.n_inference_steps).to(device)
    
    print("\nComputing test metrics ...")
    test_metrics = compute_test_metrics(flow_matcher, test_loader, device)

    print("\n=== Model RMSD ===")
    print(f"RMSD median:   {test_metrics['rmsd_median']:.3f} Å")
    print(f"RMSD mean:     {test_metrics['rmsd_mean']:.3f} Å")
    print(f"RMSD < 1 Å:    {test_metrics['rmsd_pct_under_1A']:.1f}%")
    print(f"RMSD < 2 Å:    {test_metrics['rmsd_pct_under_2A']:.1f}%")
    print(f"RMSD < 5 Å:    {test_metrics['rmsd_pct_under_5A']:.1f}%")

if __name__ == "__main__":
    main()
