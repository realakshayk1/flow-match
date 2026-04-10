"""
PyG Dataset and DataLoader for PDBBind HeteroData .pt files.
"""

import os
import json
from typing import List, Optional

import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader


class PDBBindDataset(Dataset):
    """
    Loads precomputed HeteroData .pt files from processed_dir.

    Each .pt file contains a HeteroData with:
      data['ligand'].x          [N_lig, 17]
      data['ligand'].pos        [N_lig, 3]   crystal coords (flow target x_1)
      data['ligand'].edge_index [2, E_lig]
      data['ligand'].edge_attr  [E_lig, 6]
      data['pocket'].x          [M_poc, 29]
      data['pocket'].pos        [M_poc, 3]
      data['pocket'].edge_index [2, E_poc]
      data['pocket'].edge_attr  [E_poc, 16]
      data['pocket','to','ligand'].edge_index [2, N_lig*k]
      data['pocket','to','ligand'].edge_attr  [N_lig*k, 16]
      data.complex_id           str

    Args:
        processed_dir: directory containing <complex_id>.pt files
        complex_ids:   list of complex IDs to include
    """

    def __init__(self, processed_dir: str, complex_ids: List[str]):
        super().__init__()
        self._data_dir = processed_dir  # avoid collision with PyG's processed_dir property
        self.complex_ids = [
            cid for cid in complex_ids
            if os.path.exists(os.path.join(processed_dir, f"{cid}.pt"))
        ]

    def len(self) -> int:
        return len(self.complex_ids)

    def get(self, idx: int) -> HeteroData:
        cid = self.complex_ids[idx]
        path = os.path.join(self._data_dir, f"{cid}.pt")
        data = torch.load(path, weights_only=False)
        return data


def make_dataloader(
    dataset: PDBBindDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Returns a PyG DataLoader for HeteroData batches.

    num_workers=0 on Windows (no fork support).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=["ligand__pos"],
    )


def load_splits(splits_path: str) -> dict:
    """
    Load splits.json → {'train': [...], 'val': [...], 'test': [...]}.
    """
    with open(splits_path, "r") as f:
        return json.load(f)
