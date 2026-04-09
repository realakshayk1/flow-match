"""
Pure featurization functions for ligands and protein pockets.
No PyG dependency — callers assemble graph objects from the returned tensors.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import MDAnalysis

# ---------------------------------------------------------------------------
# Atom / element vocabularies
# ---------------------------------------------------------------------------

ATOM_TYPES: List[str] = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other"]

POCKET_ELEMENTS: List[str] = ["C", "N", "O", "S", "P", "Se", "Mg", "other"]

RESIDUE_TYPES: List[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

METAL_ELEMENTS = {
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V",
    "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "RB", "SR",
    "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "IN", "SN", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM",
    "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
    "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB",
    "BI", "PO", "FR", "RA",
}

BACKBONE_ATOMS = {"CA", "C", "N", "O"}

# ---------------------------------------------------------------------------
# Ligand featurization
# ---------------------------------------------------------------------------

def featurize_ligand_atom(atom: Chem.Atom) -> Tensor:
    """
    Returns 17-dim float tensor:
      [0:10]  atom_type_onehot (ATOM_TYPES, last = 'other')
      [10]    formal_charge / 4.0
      [11:14] hybridization_onehot (SP, SP2, SP3)
      [14]    is_aromatic
      [15]    degree / 6.0
      [16]    in_ring
    """
    feats = []

    # Atom type one-hot (10 dims)
    symbol = atom.GetSymbol()
    atom_oh = [0.0] * len(ATOM_TYPES)
    idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else ATOM_TYPES.index("other")
    atom_oh[idx] = 1.0
    feats.extend(atom_oh)

    # Formal charge
    feats.append(atom.GetFormalCharge() / 4.0)

    # Hybridization one-hot (SP, SP2, SP3)
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.rdchem.HybridizationType.SP:  [1.0, 0.0, 0.0],
        Chem.rdchem.HybridizationType.SP2: [0.0, 1.0, 0.0],
        Chem.rdchem.HybridizationType.SP3: [0.0, 0.0, 1.0],
    }
    feats.extend(hyb_map.get(hyb, [0.0, 0.0, 0.0]))

    # Aromatic
    feats.append(float(atom.GetIsAromatic()))

    # Degree
    feats.append(atom.GetDegree() / 6.0)

    # In ring
    feats.append(float(atom.IsInRing()))

    return torch.tensor(feats, dtype=torch.float32)


def featurize_ligand_bond(bond: Chem.Bond) -> Tensor:
    """
    Returns 6-dim float tensor:
      [0:4]  bond_type_onehot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
      [4]    is_conjugated
      [5]    in_ring
    """
    feats = []

    bt = bond.GetBondType()
    bt_map = {
        Chem.rdchem.BondType.SINGLE:    [1.0, 0.0, 0.0, 0.0],
        Chem.rdchem.BondType.DOUBLE:    [0.0, 1.0, 0.0, 0.0],
        Chem.rdchem.BondType.TRIPLE:    [0.0, 0.0, 1.0, 0.0],
        Chem.rdchem.BondType.AROMATIC:  [0.0, 0.0, 0.0, 1.0],
    }
    feats.extend(bt_map.get(bt, [0.0, 0.0, 0.0, 0.0]))

    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))

    return torch.tensor(feats, dtype=torch.float32)


def featurize_ligand(mol: Chem.Mol) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Featurize a ligand molecule.

    Returns:
        node_feat   [N, 17]   atom features
        edge_index  [2, E]    bidirectional bond indices
        edge_feat   [E, 6]    bond features (both directions)
        pos         [N, 3]    3D coordinates from conformer

    Raises:
        ValueError if mol has no conformer.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D conformer.")

    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # [N, 3]

    node_feats = torch.stack(
        [featurize_ligand_atom(atom) for atom in mol.GetAtoms()]
    )  # [N, 17]

    src_list, dst_list, edge_feat_list = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = featurize_ligand_bond(bond)
        # Both directions
        src_list.extend([i, j])
        dst_list.extend([j, i])
        edge_feat_list.extend([feat, feat])

    if edge_feat_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_feat = torch.stack(edge_feat_list)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feat = torch.zeros((0, 6), dtype=torch.float32)

    return node_feats, edge_index, edge_feat, pos


# ---------------------------------------------------------------------------
# RBF encoding (shared)
# ---------------------------------------------------------------------------

def rbf_encoding(
    distances: Tensor,
    n_basis: int = 16,
    d_min: float = 0.0,
    d_max: float = 10.0,
) -> Tensor:
    """
    Radial basis function encoding.

    centers = linspace(d_min, d_max, n_basis)
    gamma   = (n_basis / (d_max - d_min))^2
    output  = exp(-gamma * (d - center)^2)

    Args:
        distances: [E] tensor of distances
        n_basis:   number of RBF centers
        d_min:     minimum distance for center placement
        d_max:     maximum distance for center placement

    Returns:
        [E, n_basis] tensor
    """
    distances = distances.clamp(min=1e-3)
    centers = torch.linspace(d_min, d_max, n_basis, device=distances.device)
    gamma = (n_basis / (d_max - d_min)) ** 2
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)  # [E, n_basis]
    return torch.exp(-gamma * diff.pow(2))


# ---------------------------------------------------------------------------
# Pocket featurization
# ---------------------------------------------------------------------------

def featurize_pocket_atom(atom: "MDAnalysis.core.groups.Atom") -> Tensor:
    """
    Returns 29-dim float tensor:
      [0:8]   element_onehot (POCKET_ELEMENTS)
      [8]     is_backbone (atom name in BACKBONE_ATOMS)
      [9:29]  residue_type_onehot (20 standard AA, 'other' maps to zeros)
    """
    feats = []

    # Element one-hot (8 dims)
    elem = atom.element.capitalize() if hasattr(atom, 'element') and atom.element else "other"
    elem_oh = [0.0] * len(POCKET_ELEMENTS)
    idx = (
        POCKET_ELEMENTS.index(elem)
        if elem in POCKET_ELEMENTS
        else POCKET_ELEMENTS.index("other")
    )
    elem_oh[idx] = 1.0
    feats.extend(elem_oh)

    # Is backbone
    feats.append(float(atom.name.strip() in BACKBONE_ATOMS))

    # Residue type one-hot (20 dims, zeros for non-standard)
    resname = atom.resname.strip().upper() if hasattr(atom, 'resname') else "UNK"
    res_oh = [0.0] * len(RESIDUE_TYPES)
    if resname in RESIDUE_TYPES:
        res_oh[RESIDUE_TYPES.index(resname)] = 1.0
    feats.extend(res_oh)

    return torch.tensor(feats, dtype=torch.float32)


def featurize_pocket(
    pocket_atoms: "MDAnalysis.core.groups.AtomGroup",
    dist_cutoff: float = 6.0,
    n_rbf: int = 16,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Featurize protein pocket atoms.

    Args:
        pocket_atoms: MDAnalysis AtomGroup of pocket atoms
        dist_cutoff:  max pairwise distance (Å) to add an edge
        n_rbf:        number of RBF basis functions

    Returns:
        node_feat   [M, 29]
        edge_index  [2, E']
        edge_feat   [E', 16]
        pos         [M, 3]
    """
    positions = torch.tensor(pocket_atoms.positions, dtype=torch.float32)  # [M, 3]
    M = positions.shape[0]

    node_feats = torch.stack(
        [featurize_pocket_atom(atom) for atom in pocket_atoms]
    )  # [M, 29]

    # Build edges: pairs within dist_cutoff
    if M > 1:
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [M, M, 3]
        dists = diff.norm(dim=-1)  # [M, M]
        mask = (dists < dist_cutoff) & (dists > 0)  # exclude self-loops
        src, dst = mask.nonzero(as_tuple=True)
        edge_dists = dists[src, dst]
        edge_index = torch.stack([src, dst], dim=0)
        edge_feat = rbf_encoding(edge_dists, n_basis=n_rbf, d_max=dist_cutoff)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feat = torch.zeros((0, n_rbf), dtype=torch.float32)

    return node_feats, edge_index, edge_feat, positions


# ---------------------------------------------------------------------------
# Cross edges: pocket → ligand
# ---------------------------------------------------------------------------

def build_cross_edges(
    lig_pos: Tensor,
    poc_pos: Tensor,
    k: int = 4,
    n_rbf: int = 16,
    d_max: float = 10.0,
) -> Tuple[Tensor, Tensor]:
    """
    For each ligand atom, find k nearest pocket atoms (pocket → ligand direction).

    Args:
        lig_pos: [N, 3] ligand atom positions
        poc_pos: [M, 3] pocket atom positions
        k:       number of nearest pocket atoms per ligand atom
        n_rbf:   RBF basis count
        d_max:   max distance for RBF encoding

    Returns:
        cross_edge_index  [2, N*k]  row=poc_local_idx, col=lig_local_idx
        cross_edge_attr   [N*k, 16] RBF-encoded distances

    Note:
        Indices are in local space (poc: 0..M-1, lig: 0..N-1).
        Caller must offset pocket indices by N_lig when building combined graph.
    """
    N = lig_pos.shape[0]
    M = poc_pos.shape[0]
    k_actual = min(k, M)

    # Pairwise distances: [N, M]
    diff = lig_pos.unsqueeze(1) - poc_pos.unsqueeze(0)  # [N, M, 3]
    dists = diff.norm(dim=-1)  # [N, M]

    # k-NN: for each ligand atom, k closest pocket atoms
    knn_dists, knn_idx = dists.topk(k_actual, dim=1, largest=False)  # [N, k]

    lig_idx = torch.arange(N, device=lig_pos.device).unsqueeze(1).expand(N, k_actual)
    poc_idx = knn_idx  # [N, k]

    cross_edge_index = torch.stack([
        poc_idx.reshape(-1),   # pocket source (local)
        lig_idx.reshape(-1),   # ligand target (local)
    ], dim=0)  # [2, N*k]

    cross_edge_attr = rbf_encoding(
        knn_dists.reshape(-1), n_basis=n_rbf, d_max=d_max
    )  # [N*k, 16]

    return cross_edge_index, cross_edge_attr


# ---------------------------------------------------------------------------
# Molecular filters
# ---------------------------------------------------------------------------

def passes_filters(mol: Chem.Mol) -> Tuple[bool, dict]:
    """
    Check if a ligand molecule passes data quality filters.

    Filters:
      - Molecular weight < 500 Da
      - Rotatable bonds <= 7
      - Heavy atoms <= 30
      - No metal atoms

    Returns:
        (passed: bool, details: dict with per-filter results and values)
    """
    details = {}

    mw = Descriptors.ExactMolWt(mol)
    details["mw"] = mw
    details["mw_ok"] = mw < 500.0

    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    details["rot_bonds"] = rot_bonds
    details["rot_bonds_ok"] = rot_bonds <= 7

    heavy_atoms = mol.GetNumHeavyAtoms()
    details["heavy_atoms"] = heavy_atoms
    details["heavy_atoms_ok"] = heavy_atoms <= 30

    has_metal = any(
        atom.GetSymbol().upper() in METAL_ELEMENTS for atom in mol.GetAtoms()
    )
    details["has_metal"] = has_metal
    details["no_metal_ok"] = not has_metal

    passed = all([
        details["mw_ok"],
        details["rot_bonds_ok"],
        details["heavy_atoms_ok"],
        details["no_metal_ok"],
    ])

    return passed, details
