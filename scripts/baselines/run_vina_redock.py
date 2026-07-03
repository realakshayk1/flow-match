"""
Re-dock each complex's own ligand back into its own receptor with smina/AutoDock Vina, and write
the top pose as ``{out_dir}/{cid}.sdf`` — the ``--pred_dir`` for
``scripts/baselines/score_external_poses.py``.

This is the apples-to-apples classical baseline for the pocket-conditioned re-docking task:
the search box is auto-set around the crystal ligand (``--autobox_ligand``), i.e. the binding
site is given, exactly as the Flow-Match model is given the pocket.

Requires ``smina`` (preferred) or ``vina`` on PATH. On Colab/Linux:
    conda install -c conda-forge -y smina openbabel      # or: pip install vina meeko

Example:
  python scripts/baselines/run_vina_redock.py \
      --splits data/splits_clean.json --split test \
      --raw_dir data/raw/refined-set --out_dir results/phase3/vina_poses \
      --engine smina --exhaustiveness 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _first_pose(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def _run_smina(receptor, ligand, out_sdf, exhaustiveness, num_modes, autobox_add):
    cmd = [
        "smina", "--receptor", receptor, "--ligand", ligand,
        "--autobox_ligand", ligand, "--autobox_add", str(autobox_add),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
        "--out", out_sdf, "--seed", "42",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits_clean.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--raw_dir", default="data/raw/refined-set")
    p.add_argument("--out_dir", default="results/phase3/vina_poses")
    p.add_argument("--engine", default="smina", choices=["smina"],
                   help="smina (uses --autobox_ligand). Plain vina needs an explicit box; "
                        "use smina for re-docking.")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num_modes", type=int, default=9)
    p.add_argument("--autobox_add", type=float, default=4.0)
    p.add_argument("--max_complexes", type=int, default=None)
    args = p.parse_args()

    if shutil.which(args.engine) is None:
        raise SystemExit(f"'{args.engine}' not found on PATH. Install it (see module docstring).")

    with open(args.splits) as f:
        ids = json.load(f)[args.split]
    if args.max_complexes:
        ids = ids[: args.max_complexes]

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok, n_fail = 0, 0
    for i, cid in enumerate(ids):
        receptor = os.path.join(args.raw_dir, cid, f"{cid}_protein.pdb")
        ligand = os.path.join(args.raw_dir, cid, f"{cid}_ligand.sdf")
        if not (os.path.exists(receptor) and os.path.exists(ligand)):
            n_fail += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            out_sdf = os.path.join(td, "docked.sdf")
            try:
                _run_smina(receptor, ligand, out_sdf, args.exhaustiveness,
                           args.num_modes, args.autobox_add)
                pose = _first_pose(out_sdf) if os.path.exists(out_sdf) else None
            except subprocess.CalledProcessError:
                pose = None
            if pose is None:
                n_fail += 1
                continue
            w = Chem.SDWriter(os.path.join(args.out_dir, f"{cid}.sdf"))
            w.write(pose)
            w.close()
            n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)} done ({n_ok} ok, {n_fail} fail)")

    print(f"Docked {n_ok} complexes -> {args.out_dir} (failed: {n_fail}) of {len(ids)}")


if __name__ == "__main__":
    main()
