"""
Blind docking via P2Rank + Flow-Match model.

Predicts binding pocket centroids using P2Rank, extracts a 10 Å point cloud
around each centroid (same as training), runs the trained FlowMatcher on each,
scores poses by MMFF94 strain energy, and outputs the best pose as SDF.

Usage:
    python scripts/blind_dock.py \
        --protein   receptor.pdb \
        --ligand    ligand.sdf \
        --checkpoint checkpoints/best_model.pt \
        --out       blind_dock_out.sdf \
        --n_pockets 3 \
        --n_steps   20 \
        --crystal   crystal.sdf  # optional; enables RMSD reporting
        --p2rank    ./p2rank_2.4/prank

Colab P2Rank setup:
    !wget -q https://github.com/rdk/p2rank/releases/download/2.4/p2rank_2.4.tar.gz
    !tar -xzf p2rank_2.4.tar.gz
    # executable: ./p2rank_2.4/prank
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import MDAnalysis as mda

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.featurize import (
    featurize_ligand,
    featurize_pocket,
    build_cross_edges,
)
from src.models.egnn import build_default_model
from src.models.flow_model import FlowMatcher
from src.training.metrics import kabsch_rmsd, mmff94_energy


# ---------------------------------------------------------------------------
# P2Rank helpers
# ---------------------------------------------------------------------------

def run_p2rank(p2rank_exe: str, protein_pdb: str, out_dir: str) -> str:
    """Run P2Rank and return path to the predictions CSV."""
    cmd = [p2rank_exe, "predict", "-f", protein_pdb, "-o", out_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"P2Rank failed:\n{result.stderr}")

    pdb_name = os.path.splitext(os.path.basename(protein_pdb))[0]
    csv_path = os.path.join(out_dir, f"{pdb_name}.pdb_predictions.csv")
    if not os.path.exists(csv_path):
        # Some versions use a slightly different naming convention
        candidates = [f for f in os.listdir(out_dir) if f.endswith("_predictions.csv")]
        if not candidates:
            raise FileNotFoundError(
                f"P2Rank predictions CSV not found in {out_dir}.\n"
                f"P2Rank stdout:\n{result.stdout}"
            )
        csv_path = os.path.join(out_dir, candidates[0])
    return csv_path


def parse_p2rank_csv(csv_path: str, n_pockets: int) -> list:
    """
    Parse P2Rank predictions CSV.
    Returns list of (rank, score, cx, cy, cz) for top n_pockets.
    """
    pockets = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys/values (P2Rank adds spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}
            rank = int(row.get("rank", len(pockets) + 1))
            score = float(row.get("score", 0.0))
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            cz = float(row["center_z"])
            pockets.append((rank, score, cx, cy, cz))
            if len(pockets) >= n_pockets:
                break
    return pockets


# ---------------------------------------------------------------------------
# Ligand loading
# ---------------------------------------------------------------------------

def load_ligand(sdf_path: str = None, smiles: str = None) -> Chem.Mol:
    """Load ligand from SDF or SMILES. Embeds a conformer for featurization."""
    if sdf_path:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
        mol = next(m for m in suppl if m is not None)
    elif smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mol = Chem.RWMol(mol)
        Chem.SanitizeMol(mol)
        mol = mol.GetMol()
    else:
        raise ValueError("Provide either --ligand or --smiles")

    # Ensure a 3D conformer exists (featurize_ligand requires one)
    if mol.GetNumConformers() == 0:
        mol_h = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if ret != 0:
            raise RuntimeError("ETKDG embedding failed — cannot generate conformer for ligand")
        mol = Chem.RemoveHs(mol_h)

    return mol


# ---------------------------------------------------------------------------
# Pocket extraction + graph construction
# ---------------------------------------------------------------------------

def extract_pocket_graph(
    universe: mda.Universe,
    centroid: tuple,
    lig_pos: torch.Tensor,
    pocket_cutoff: float = 10.0,
    cross_k: int = 4,
    n_rbf: int = 16,
    min_atoms: int = 10,
):
    """
    Extract pocket atoms within pocket_cutoff Å of centroid and build graph tensors.
    Returns None if too few atoms are found.
    """
    cx, cy, cz = centroid
    sel = f"protein and (point {cx:.3f} {cy:.3f} {cz:.3f} {pocket_cutoff})"
    pocket_atoms = universe.select_atoms(sel)

    if len(pocket_atoms) < min_atoms:
        return None

    poc_h, poc_edge_index, poc_edge_attr, poc_pos = featurize_pocket(
        pocket_atoms, dist_cutoff=6.0, n_rbf=n_rbf
    )
    cross_edge_index, cross_edge_attr = build_cross_edges(
        lig_pos, poc_pos, k=cross_k, n_rbf=n_rbf
    )

    return poc_h, poc_edge_index, poc_edge_attr, poc_pos, cross_edge_index, cross_edge_attr


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def dock_one_pocket(
    flow_matcher: FlowMatcher,
    lig_h: torch.Tensor,
    lig_edge_index: torch.Tensor,
    lig_edge_attr: torch.Tensor,
    poc_h: torch.Tensor,
    poc_pos: torch.Tensor,
    poc_edge_index: torch.Tensor,
    poc_edge_attr: torch.Tensor,
    cross_edge_index: torch.Tensor,
    cross_edge_attr: torch.Tensor,
    device: torch.device,
    n_steps: int,
) -> np.ndarray:
    """Run inference for one pocket. Returns [N_lig, 3] numpy array in PDB frame."""
    # Center pocket (matches training convention)
    poc_center = poc_pos.mean(dim=0)
    poc_pos_c = poc_pos - poc_center

    pred = flow_matcher.generate_single(
        lig_h=lig_h.to(device),
        poc_x=poc_pos_c.to(device),
        poc_h=poc_h.to(device),
        lig_edge_index=lig_edge_index.to(device),
        lig_edge_attr=lig_edge_attr.to(device),
        poc_edge_index=poc_edge_index.to(device),
        poc_edge_attr=poc_edge_attr.to(device),
        cross_edge_index=cross_edge_index.to(device),
        cross_edge_attr=cross_edge_attr.to(device),
        n_steps=n_steps,
    )

    # Shift back to original PDB frame (generate_single returns centered coords)
    pred_world = pred + poc_center.to(device)
    return pred_world.cpu().numpy()


# ---------------------------------------------------------------------------
# SDF output
# ---------------------------------------------------------------------------

def write_sdf(mol: Chem.Mol, coords: np.ndarray, out_path: str):
    """Write mol with generated coords to SDF."""
    mol_copy = Chem.RWMol(mol)
    if mol_copy.GetNumConformers() == 0:
        conf = Chem.Conformer(mol_copy.GetNumAtoms())
        mol_copy.AddConformer(conf, assignId=True)
    conf = mol_copy.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    writer = Chem.SDWriter(out_path)
    writer.write(mol_copy.GetMol())
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Blind docking via P2Rank + Flow-Match")
    parser.add_argument("--protein",    required=True, help="Protein PDB file")
    parser.add_argument("--ligand",     default=None,  help="Ligand SDF file")
    parser.add_argument("--smiles",     default=None,  help="Ligand SMILES string")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--out",        default="blind_dock_out.sdf", help="Output SDF path")
    parser.add_argument("--crystal",    default=None,  help="Crystal ligand SDF for RMSD eval")
    parser.add_argument("--n_pockets",  type=int, default=3, help="Top-K P2Rank pockets to try")
    parser.add_argument("--n_steps",    type=int, default=20, help="ODE Euler steps")
    parser.add_argument("--pocket_cutoff", type=float, default=10.0, help="Pocket radius (Å)")
    parser.add_argument("--cross_k",    type=int, default=4, help="Cross k-NN neighbours")
    parser.add_argument("--p2rank",     default="prank", help="Path to P2Rank executable")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = ckpt.get("run_config", {})
    hidden_dim = run_config.get("hidden_dim", 128)
    n_layers   = run_config.get("n_layers", 6)
    model = build_default_model(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    flow_matcher = FlowMatcher(model, n_steps=args.n_steps).to(device)
    print(f"Model loaded (hidden_dim={hidden_dim}, n_layers={n_layers})")

    # --- Load ligand ---
    mol = load_ligand(sdf_path=args.ligand, smiles=args.smiles)
    lig_h, lig_edge_index, lig_edge_attr, lig_pos = featurize_ligand(mol)
    print(f"Ligand: {mol.GetNumHeavyAtoms()} heavy atoms")

    # --- Load crystal (optional) ---
    crystal_coords = None
    if args.crystal:
        suppl = Chem.SDMolSupplier(args.crystal, removeHs=True)
        crystal_mol = next(m for m in suppl if m is not None)
        crystal_coords = torch.tensor(
            crystal_mol.GetConformer().GetPositions(), dtype=torch.float32
        )
        print(f"Crystal pose loaded for RMSD evaluation")

    # --- Run P2Rank ---
    print(f"\nRunning P2Rank on {args.protein} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = run_p2rank(args.p2rank, args.protein, tmpdir)
        pockets = parse_p2rank_csv(csv_path, args.n_pockets)

    print(f"P2Rank found {len(pockets)} pocket(s) to evaluate\n")

    # --- Load protein for MDAnalysis ---
    universe = mda.Universe(args.protein)

    # --- Dock each pocket ---
    results = []
    for rank, p2rank_score, cx, cy, cz in pockets:
        print(f"  Pocket {rank} (P2Rank score={p2rank_score:.1f}, center=({cx:.1f},{cy:.1f},{cz:.1f}))")

        graph = extract_pocket_graph(
            universe, (cx, cy, cz), lig_pos,
            pocket_cutoff=args.pocket_cutoff,
            cross_k=args.cross_k,
        )
        if graph is None:
            print(f"    → Skipped: fewer than 10 pocket atoms within {args.pocket_cutoff} Å")
            continue

        poc_h, poc_edge_index, poc_edge_attr, poc_pos, cross_edge_index, cross_edge_attr = graph

        with torch.no_grad():
            pred_coords = dock_one_pocket(
                flow_matcher,
                lig_h, lig_edge_index, lig_edge_attr,
                poc_h, poc_pos, poc_edge_index, poc_edge_attr,
                cross_edge_index, cross_edge_attr,
                device, args.n_steps,
            )

        # Score by MMFF94 strain energy
        strain, strain_fail = mmff94_energy(mol, pred_coords)
        strain_str = f"{strain:.1f} kcal/mol" if strain is not None else f"N/A ({strain_fail})"

        # RMSD vs crystal (if provided)
        rmsd = None
        if crystal_coords is not None:
            pred_t = torch.tensor(pred_coords, dtype=torch.float32)
            rmsd = kabsch_rmsd(pred_t, crystal_coords)

        rmsd_str = f"{rmsd:.3f} Å" if rmsd is not None else "N/A"
        print(f"    strain={strain_str} | RMSD={rmsd_str}")

        results.append({
            "rank": rank,
            "p2rank_score": p2rank_score,
            "coords": pred_coords,
            "strain": strain if strain is not None else float("inf"),
            "rmsd": rmsd,
        })

    if not results:
        print("\nNo valid pockets found. Check P2Rank output and protein PDB format.")
        sys.exit(1)

    # --- Pick best pose by strain energy ---
    best = min(results, key=lambda r: r["strain"])
    print(f"\nBest pose: pocket {best['rank']} (strain={best['strain']:.1f} kcal/mol"
          + (f", RMSD={best['rmsd']:.3f} Å" if best["rmsd"] is not None else "") + ")")

    write_sdf(mol, best["coords"], args.out)
    print(f"Saved to {args.out}")

    # --- Summary table ---
    print("\n--- All pockets ---")
    header = f"{'Pocket':>7}  {'P2Rank':>8}  {'Strain (kcal/mol)':>18}  {'RMSD':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        rmsd_col = f"{r['rmsd']:.3f} Å" if r["rmsd"] is not None else "    N/A"
        strain_col = f"{r['strain']:.1f}" if r["strain"] != float("inf") else "    N/A"
        marker = " ← best" if r is best else ""
        print(f"{r['rank']:>7}  {r['p2rank_score']:>8.1f}  {strain_col:>18}  {rmsd_col:>8}{marker}")


if __name__ == "__main__":
    main()
