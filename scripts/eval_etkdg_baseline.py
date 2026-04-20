import argparse
import sys
import os

import torch
from rdkit import Chem

# ensure src is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import PDBBindDataset, load_splits, make_dataloader
from src.training.metrics import compute_etkdg_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    splits = load_splits(args.splits)
    test_ds  = PDBBindDataset(args.processed_dir, splits["test"])
    test_loader  = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    mol_lookup = {}
    for data in test_ds:
        smiles = getattr(data, "ligand_meta_canonical_smiles", None)
        if not smiles and hasattr(data, "smiles"):
            smiles = data.smiles
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_lookup[data.complex_id] = mol

    print(f"\nComputing ETKDG baseline on {len(mol_lookup)} molecules...")
    etkdg_metrics = compute_etkdg_baseline(test_loader, mol_lookup)

    print("\n=== ETKDG Baseline ===")
    print(f"RMSD median:   {etkdg_metrics['rmsd_median']:.3f} Å")
    print(f"RMSD mean:     {etkdg_metrics['rmsd_mean']:.3f} Å")
    print(f"RMSD < 1 Å:    {etkdg_metrics['rmsd_pct_under_1A']:.1f}%")
    print(f"RMSD < 2 Å:    {etkdg_metrics['rmsd_pct_under_2A']:.1f}%")
    print(f"RMSD < 5 Å:    {etkdg_metrics['rmsd_pct_under_5A']:.1f}%")
    print(f"\nFailures breakdown: {etkdg_metrics.get('failures', {})}")

if __name__ == "__main__":
    main()
