import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_first_mol(sdf_path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Failed to parse SDF: {sdf_path}")
    return mol


def mol_coords(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)


def write_pose_sdf(template_mol: Chem.Mol, coords: np.ndarray, out_path: Path) -> None:
    mol_copy = Chem.RWMol(template_mol)
    if mol_copy.GetNumConformers() == 0:
        conf = Chem.Conformer(mol_copy.GetNumAtoms())
        mol_copy.AddConformer(conf, assignId=True)
    conf = mol_copy.GetConformer()
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    ensure_dir(out_path.parent)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol_copy.GetMol())
    writer.close()


def copy_pose_coords_onto_template(template: Chem.Mol, pose: Chem.Mol) -> Chem.Mol:
    """
    Return a copy of `template` with 3D coordinates taken from `pose`.

    Assumes identical atom count and compatible topology (typical when obabel
    converts Vina output that was built from the same ligand graph).
    """
    if template.GetNumAtoms() != pose.GetNumAtoms():
        raise ValueError(
            f"atom count mismatch: template={template.GetNumAtoms()} pose={pose.GetNumAtoms()}"
        )
    mol = Chem.Mol(template)
    conf_m = mol.GetConformer()
    conf_p = pose.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf_m.SetAtomPosition(i, conf_p.GetAtomPosition(i))
    return mol


def rmsd_kabsch_from_mols(ref_mol: Chem.Mol, pred_mol: Chem.Mol) -> float:
    from src.training.metrics import kabsch_rmsd

    ref = torch.tensor(mol_coords(ref_mol), dtype=torch.float32)
    pred = torch.tensor(mol_coords(pred_mol), dtype=torch.float32)
    if ref.shape[0] != pred.shape[0]:
        return float("nan")
    return float(kabsch_rmsd(pred, ref))
