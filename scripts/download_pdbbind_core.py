"""
Download the PDBBind 2016 core set (public, 290 complexes) as a dev fallback.

The core set requires registration at http://www.pdbbind.org.cn to download.
This script provides instructions and a helper to extract the archive once
you have downloaded it manually.

Usage (after manual download):
    python scripts/download_pdbbind_core.py --archive /path/to/CASF-2016.tar.gz
    python scripts/download_pdbbind_core.py --archive /path/to/coreset.tar.gz

Or for testing with a tiny synthetic dataset:
    python scripts/download_pdbbind_core.py --synthetic --n 10
"""

import argparse
import os
import sys
import tarfile
import shutil


DOWNLOAD_INSTRUCTIONS = """
PDBBind 2016 Core Set Download Instructions
============================================

The PDBBind core set is not freely downloadable without registration.

1. Register at: http://www.pdbbind.org.cn/register.php
2. Download the "CASF-2016" package from the download page.
   File is typically named: CASF-2016.tar.gz (~2 GB)
3. Run this script with the downloaded archive:
   python scripts/download_pdbbind_core.py --archive /path/to/CASF-2016.tar.gz

Alternatively, for a small subset to test your pipeline, individual PDB entries
can be fetched from the RCSB PDB:
   python scripts/download_pdbbind_core.py --rcsb_ids 3eml 1a28 1d3p

Expected directory structure after extraction:
    data/raw/
        1a28/
            1a28_protein.pdb
            1a28_ligand.sdf
        3eml/
            3eml_protein.pdb
            3eml_ligand.sdf
        ...
"""


def extract_archive(archive_path: str, raw_dir: str):
    """Extract a PDBBind archive to data/raw/."""
    print(f"Extracting {archive_path} → {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(raw_dir)
    print("Done. Check data/raw/ for extracted complexes.")


def download_rcsb_subset(pdb_ids: list, raw_dir: str):
    """
    Download individual PDB structures from RCSB as a tiny test subset.
    Downloads protein PDB only — ligand SDF must be sourced from PDBBind.
    """
    try:
        import urllib.request
    except ImportError:
        print("urllib not available")
        return

    os.makedirs(raw_dir, exist_ok=True)
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()
        out_dir = os.path.join(raw_dir, pdb_id)
        os.makedirs(out_dir, exist_ok=True)
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        out_path = os.path.join(out_dir, f"{pdb_id}_protein.pdb")
        if os.path.exists(out_path):
            print(f"  {pdb_id}: already downloaded")
            continue
        try:
            print(f"  Downloading {pdb_id} from RCSB...")
            urllib.request.urlretrieve(url, out_path)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  Failed {pdb_id}: {e}")


def make_synthetic_dataset(n: int, raw_dir: str):
    """
    Create a tiny synthetic dataset for pipeline testing.
    Uses RDKit to generate random small molecules with 3D conformers.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np
    except ImportError:
        print("RDKit not available. Install the conda environment first.")
        return

    os.makedirs(raw_dir, exist_ok=True)

    # Simple molecules for testing
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
        "c1ccc2ccccc2c1",  # naphthalene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "O=C(O)c1ccccc1O",  # salicylic acid
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
        "OC(=O)c1ccc(Cl)cc1",  # 4-chlorobenzoic acid
        "c1cnc2ccccc2n1",  # phthalazine
        "O=C(O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
    ]

    print(f"Generating {n} synthetic complexes in {raw_dir}")
    for i in range(min(n, len(smiles_list))):
        smi = smiles_list[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol, randomSeed=i)
        if ret != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        pdb_id = f"syn{i:04d}"
        out_dir = os.path.join(raw_dir, pdb_id)
        os.makedirs(out_dir, exist_ok=True)

        # Save ligand SDF
        sdf_path = os.path.join(out_dir, f"{pdb_id}_ligand.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        # Generate a synthetic protein PDB (just a few backbone atoms around centroid)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        centroid = pos.mean(axis=0)
        _write_synthetic_protein(pdb_id, centroid, out_dir)

        print(f"  Created {pdb_id} ({mol.GetNumHeavyAtoms()} atoms)")


def _write_synthetic_protein(pdb_id: str, centroid, out_dir: str):
    """Write a minimal synthetic protein PDB with backbone atoms around centroid."""
    import numpy as np

    np.random.seed(abs(hash(pdb_id)) % 2**31)
    n_residues = 30
    lines = ["REMARK synthetic protein for pipeline testing"]

    atom_idx = 1
    residue_names = ["ALA", "GLY", "VAL", "LEU", "SER"] * (n_residues // 5 + 1)

    for res_i in range(n_residues):
        res_pos = centroid + np.random.randn(3) * 8.0
        for atom_name in ["N", "CA", "C", "O"]:
            atom_pos = res_pos + np.random.randn(3) * 0.5
            chain = "A"
            res_num = res_i + 1
            resname = residue_names[res_i]
            element = atom_name[0]
            line = (
                f"ATOM  {atom_idx:5d}  {atom_name:<3s} {resname} {chain}"
                f"{res_num:4d}    "
                f"{atom_pos[0]:8.3f}{atom_pos[1]:8.3f}{atom_pos[2]:8.3f}"
                f"  1.00  0.00           {element}"
            )
            lines.append(line)
            atom_idx += 1

    lines.append("END")
    pdb_path = os.path.join(out_dir, f"{pdb_id}_protein.pdb")
    with open(pdb_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download/prepare PDBBind data")
    parser.add_argument("--archive", help="Path to downloaded PDBBind archive (.tar.gz)")
    parser.add_argument("--rcsb_ids", nargs="+", help="RCSB PDB IDs to download")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data")
    parser.add_argument("--n", type=int, default=10, help="Number of synthetic complexes")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    if not any([args.archive, args.rcsb_ids, args.synthetic]):
        print(DOWNLOAD_INSTRUCTIONS)
        return

    if args.archive:
        extract_archive(args.archive, args.raw_dir)

    if args.rcsb_ids:
        download_rcsb_subset(args.rcsb_ids, args.raw_dir)

    if args.synthetic:
        make_synthetic_dataset(args.n, args.raw_dir)


if __name__ == "__main__":
    main()
