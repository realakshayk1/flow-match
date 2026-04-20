import argparse
import sys
import os
import torch
from rdkit import Chem

# ensure src is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import PDBBindDataset, load_splits
from src.training.metrics import mmff94_energy, etkdg_energy, kabsch_rmsd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    splits = load_splits(args.splits)
    test_ds  = PDBBindDataset(args.processed_dir, splits["test"])
    
    print(f"Evaluating strain for {args.n_samples} samples from test set...")
    
    for i in range(min(args.n_samples, len(test_ds))):
        data = test_ds[i]
        smiles = getattr(data, "ligand_meta_canonical_smiles", None)
        if not smiles and hasattr(data, "smiles"):
            smiles = data.smiles
        if not smiles: continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        
        crystal_pos = data["ligand"].pos.cpu().numpy()
        
        try:
            # ETKDG
            mol_h = Chem.AddHs(mol)
            ret = Chem.AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if ret != 0: continue
            Chem.AllChem.MMFFOptimizeMolecule(mol_h)
            
            mol_h = Chem.RemoveHs(mol_h)
            etkdg_pos = mol_h.GetConformer().GetPositions()
        except:
            continue
            
        crys_energy, crys_reason = mmff94_energy(mol, crystal_pos)
        etkdg_en, etkdg_reason = etkdg_energy(mol)
        
        c_rmsd = kabsch_rmsd(torch.tensor(etkdg_pos, dtype=torch.float32), torch.tensor(crystal_pos, dtype=torch.float32))
        
        print(f"Complex: {data.complex_id} | Atoms: {mol.GetNumAtoms()}")
        print(f"  Crystal Energy: {crys_energy if crys_energy is not None else crys_reason} kcal/mol")
        print(f"  ETKDG Energy:   {etkdg_en if etkdg_en is not None else etkdg_reason} kcal/mol")
        print(f"  ETKDG vs Crystal RMSD: {c_rmsd:.3f} Å")
        print("-" * 40)

if __name__ == "__main__":
    main()
