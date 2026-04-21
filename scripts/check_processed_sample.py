import os
import glob
import torch

def main():
    """
    Check whether preprocess.py's atom-identity and chemical metadata 
    was properly tracked into the processed samples.
    """
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        print(f"Directory {processed_dir} not found.")
        return
        
    pt_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    if not pt_files:
        print(f"No .pt files found in {processed_dir}.")
        return
        
    print(f"Found {len(pt_files)} processed items. Checking first 5 samples...\n")
    
    for pt in pt_files[:5]:
        data = torch.load(pt, map_location="cpu", weights_only=False)
        print(f"--- File: {os.path.basename(pt)} ---")
        print(f"  Complex ID: {getattr(data, 'complex_id', 'Missing')}")
        
        if hasattr(data, 'ligand_meta_canonical_smiles'):
            print(f"  Canonical SMILES: {data.ligand_meta_canonical_smiles}")
        elif hasattr(data, 'smiles'):
            print(f"  Legacy SMILES: {data.smiles}")
        else:
            print("  SMILES: Missing")
            
        print(f"  Atom count metadata: {getattr(data, 'ligand_meta_atom_count', 'Missing')}")
        print(f"  Heavy atom count metadata: {getattr(data, 'ligand_meta_heavy_atom_count', 'Missing')}")
        
        symbols = getattr(data, 'ligand_meta_ordered_symbols', 'Missing')
        if symbols != 'Missing' and len(symbols) > 10:
            symbols_str = f"{symbols[:5]} ... {symbols[-5:]} (len: {len(symbols)})"
        else:
            symbols_str = str(symbols)
        print(f"  Ordered symbols: {symbols_str}")
        
        if hasattr(data, 'ligand') and hasattr(data['ligand'], 'pos'):
            pos = data['ligand'].pos
            print(f"  Ligand coordinate tensor shape: {pos.shape}")
        else:
            print("  Ligand pos tensor: Missing")
            
        print()

if __name__ == "__main__":
    main()
