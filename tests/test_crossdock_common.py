"""Tests for cross-docking shared helpers."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.crossdock.common import copy_pose_coords_onto_template, mol_coords


def test_copy_pose_coords_onto_template_matches_pose_geometry():
    template = Chem.MolFromSmiles("CC")
    pose = Chem.MolFromSmiles("CC")
    template = Chem.AddHs(template)
    pose = Chem.AddHs(pose)
    AllChem.EmbedMolecule(template, randomSeed=1)
    AllChem.EmbedMolecule(pose, randomSeed=2)
    conf = pose.GetConformer()
    for i in range(pose.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 10.0, p.y - 2.0, p.z + 0.5))

    merged = copy_pose_coords_onto_template(Chem.RemoveHs(template), Chem.RemoveHs(pose))
    assert np.allclose(mol_coords(merged), mol_coords(Chem.RemoveHs(pose)), atol=1e-5)


def test_copy_pose_coords_onto_template_rejects_atom_count_mismatch():
    a = Chem.MolFromSmiles("C")
    b = Chem.MolFromSmiles("CC")
    for m in (a, b):
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=0)
    a = Chem.RemoveHs(a)
    b = Chem.RemoveHs(b)
    try:
        copy_pose_coords_onto_template(a, b)
    except ValueError as exc:
        assert "atom count mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
