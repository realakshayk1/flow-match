"""
Flow matching wrapper around EGNNFlowModel.

Implements:
  - Training loss (conditional flow matching MSE)
  - Inference (20-step Euler integration)
  - Batch unpacking from PyG HeteroData
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.models.egnn import EGNNFlowModel


class FlowMatcher(nn.Module):
    """
    Flow matching wrapper around EGNNFlowModel.

    Training (conditional flow matching):
        x_0 ~ N(0, I)
        x_1  = crystal ligand coords
        t   ~ Uniform(0, 1)    (sampled per molecule in batch)
        x_t  = (1 - t) * x_0 + t * x_1
        target = x_1 - x_0    (constant velocity field)
        loss   = MSE(model(x_t, pocket, t), target)

    Inference (20-step Euler):
        x_0 ~ N(0, I)
        dt   = 1.0 / n_steps
        for t in linspace(0, 1-dt, n_steps):
            x = x + dt * model(x, pocket, t)

    Args:
        model:   EGNNFlowModel — the velocity field
        n_steps: int — Euler integration steps at inference (default 20)
    """

    def __init__(self, model: EGNNFlowModel, n_steps: int = 20):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: HeteroData) -> dict:
        """
        Extract tensors from a PyG HeteroData batch.

        PyG's DataLoader automatically offsets node indices in edge tensors
        when batching multiple HeteroData objects. For hetero edge type
        ('pocket', 'to', 'ligand'), PyG offsets pocket indices by cumulative
        pocket graph sizes and ligand indices by cumulative ligand graph sizes.
        So we can read cross_edge_index directly — no manual offset needed here.

        Returns dict of tensors and the lig_batch assignment vector.
        """
        return {
            "lig_x":    batch["ligand"].pos,
            "lig_h":    batch["ligand"].x,
            "poc_x":    batch["pocket"].pos,
            "poc_h":    batch["pocket"].x,
            "lig_edge_index": batch["ligand", "bond", "ligand"].edge_index,
            "lig_edge_attr":  batch["ligand", "bond", "ligand"].edge_attr,
            "poc_edge_index": batch["pocket", "bond", "pocket"].edge_index,
            "poc_edge_attr":  batch["pocket", "bond", "pocket"].edge_attr,
            "cross_edge_index": batch["pocket", "to", "ligand"].edge_index,
            "cross_edge_attr":  batch["pocket", "to", "ligand"].edge_attr,
            "lig_batch": batch["ligand"].batch,   # [N_total_lig] graph assignment
            "poc_batch": batch["pocket"].batch,   # [M_total_poc] graph assignment
        }

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: HeteroData) -> Tensor:
        """
        Compute flow matching MSE loss on a batch.

        t is sampled once per molecule (not per atom).  All coordinates are
        centered on the pocket centroid before the model call to prevent
        coordinate-scale explosion in EGNN cross edges (crystal PDB coords
        vs. N(0,I) noise have ~50 Å offsets that blow up dist_sq).

        Returns: scalar loss averaged over atoms.
        """
        unpacked = self._unpack_batch(batch)
        lig_x1   = unpacked["lig_x"]      # crystal coords: [N_total, 3]
        lig_batch = unpacked["lig_batch"]  # [N_total] — which graph each atom belongs to

        n_graphs = lig_batch.max().item() + 1
        device   = lig_x1.device

        # Sample t per molecule
        t_per_graph = torch.rand(n_graphs, device=device)   # [n_graphs]

        # Run one forward pass per graph (different t per molecule requires this)
        v_pred_list  = []
        target_list  = []

        for g in range(n_graphs):
            mask_lig = lig_batch == g
            mask_poc = unpacked["poc_batch"] == g

            # Crystal ligand and pocket positions for this graph
            lig_x1_g = lig_x1[mask_lig]                   # [N_lig, 3]
            poc_x_g  = unpacked["poc_x"][mask_poc]         # [M_poc, 3]

            # Center on pocket centroid so that N(0,I) noise and pocket
            # coordinates are in the same ~Angstrom-scale neighbourhood.
            poc_center = poc_x_g.mean(0, keepdim=True)     # [1, 3]
            lig_x1_g   = lig_x1_g - poc_center
            poc_x_g    = poc_x_g  - poc_center

            # Sample x_0 ~ N(0, I) in centered space and interpolate
            x0_g  = torch.randn_like(lig_x1_g)
            t_g   = t_per_graph[g]
            x_t_g = (1.0 - t_g) * x0_g + t_g * lig_x1_g

            # Target velocity: v* = x_1 - x_0  (same formula, centered space)
            target_list.append(lig_x1_g - x0_g)

            lig_h_g  = unpacked["lig_h"][mask_lig]
            poc_h_g  = unpacked["poc_h"][mask_poc]

            # Remap edge indices to per-graph local space
            lig_ei_g, lig_ea_g = _extract_subgraph_edges(
                unpacked["lig_edge_index"], unpacked["lig_edge_attr"],
                mask_lig, offset=mask_lig.nonzero(as_tuple=True)[0].min().item()
            )
            poc_ei_g, poc_ea_g = _extract_subgraph_edges(
                unpacked["poc_edge_index"], unpacked["poc_edge_attr"],
                mask_poc, offset=mask_poc.nonzero(as_tuple=True)[0].min().item()
            )
            cross_ei_g, cross_ea_g = _extract_cross_edges(
                unpacked["cross_edge_index"], unpacked["cross_edge_attr"],
                mask_lig, mask_poc,
                lig_offset=mask_lig.nonzero(as_tuple=True)[0].min().item(),
                poc_offset=mask_poc.nonzero(as_tuple=True)[0].min().item(),
            )

            v_g = self.model(
                lig_x=x_t_g, lig_h=lig_h_g,
                poc_x=poc_x_g, poc_h=poc_h_g,
                lig_edge_index=lig_ei_g, lig_edge_attr=lig_ea_g,
                poc_edge_index=poc_ei_g, poc_edge_attr=poc_ea_g,
                cross_edge_index=cross_ei_g, cross_edge_attr=cross_ea_g,
                t=t_g,
            )
            v_pred_list.append(v_g)

        v_pred = torch.cat(v_pred_list, dim=0)   # [N_total, 3]
        target = torch.cat(target_list,  dim=0)   # [N_total, 3]
        loss   = nn.functional.mse_loss(v_pred, target)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        batch: HeteroData,
        n_steps: Optional[int] = None,
    ) -> List[Tensor]:
        """
        Generate ligand conformations via 20-step Euler integration.

        x_0 ~ N(0, I)
        dt   = 1.0 / n_steps
        for t in linspace(0, 1 - dt, n_steps):
            v = model(x, pocket, t)
            x = x + dt * v
            x = _clamp_update(x, x_prev, max_delta=5.0)

        Returns:
            List of [N_lig_i, 3] tensors, one per molecule in batch.
        """
        n_steps = n_steps or self.n_steps
        unpacked = self._unpack_batch(batch)
        lig_x1   = unpacked["lig_x"]
        lig_batch = unpacked["lig_batch"]
        device   = lig_x1.device

        n_graphs = lig_batch.max().item() + 1
        dt = 1.0 / n_steps

        # We generate per-graph to handle different time steps cleanly
        results = []
        for g in range(n_graphs):
            mask_lig = lig_batch == g
            mask_poc = unpacked["poc_batch"] == g

            lig_h_g  = unpacked["lig_h"][mask_lig]
            poc_x_g  = unpacked["poc_x"][mask_poc]
            poc_h_g  = unpacked["poc_h"][mask_poc]

            # Center on pocket centroid (matches the centering done during training)
            poc_center = poc_x_g.mean(0, keepdim=True)   # [1, 3]
            poc_x_g    = poc_x_g - poc_center

            lig_ei_g, lig_ea_g = _extract_subgraph_edges(
                unpacked["lig_edge_index"], unpacked["lig_edge_attr"],
                mask_lig, offset=mask_lig.nonzero(as_tuple=True)[0].min().item()
            )
            poc_ei_g, poc_ea_g = _extract_subgraph_edges(
                unpacked["poc_edge_index"], unpacked["poc_edge_attr"],
                mask_poc, offset=mask_poc.nonzero(as_tuple=True)[0].min().item()
            )
            cross_ei_g, cross_ea_g = _extract_cross_edges(
                unpacked["cross_edge_index"], unpacked["cross_edge_attr"],
                mask_lig, mask_poc,
                lig_offset=mask_lig.nonzero(as_tuple=True)[0].min().item(),
                poc_offset=mask_poc.nonzero(as_tuple=True)[0].min().item(),
            )

            # Start from Gaussian noise in centered pocket space
            x = torch.randn(mask_lig.sum().item(), 3, device=device)

            for step in range(n_steps):
                t = torch.tensor(step * dt, device=device)
                x_prev = x.clone()

                v = self.model(
                    lig_x=x, lig_h=lig_h_g,
                    poc_x=poc_x_g, poc_h=poc_h_g,
                    lig_edge_index=lig_ei_g, lig_edge_attr=lig_ea_g,
                    poc_edge_index=poc_ei_g, poc_edge_attr=poc_ea_g,
                    cross_edge_index=cross_ei_g, cross_edge_attr=cross_ea_g,
                    t=t,
                )
                x = x + dt * v
                x = _clamp_update(x, x_prev, max_delta=5.0)

            results.append(x)

        return results

    # ------------------------------------------------------------------
    # Single-molecule generate (convenience)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_single(
        self,
        lig_h: Tensor, poc_x: Tensor, poc_h: Tensor,
        lig_edge_index: Tensor, lig_edge_attr: Tensor,
        poc_edge_index: Tensor, poc_edge_attr: Tensor,
        cross_edge_index: Tensor, cross_edge_attr: Tensor,
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate one conformation for a single unbatched molecule.

        Returns: [N_lig, 3] generated coordinates.
        """
        n_steps = n_steps or self.n_steps
        dt = 1.0 / n_steps
        device = lig_h.device

        x = torch.randn(lig_h.size(0), 3, device=device)

        for step in range(n_steps):
            t = torch.tensor(step * dt, device=device)
            x_prev = x.clone()
            v = self.model(
                lig_x=x, lig_h=lig_h,
                poc_x=poc_x, poc_h=poc_h,
                lig_edge_index=lig_edge_index, lig_edge_attr=lig_edge_attr,
                poc_edge_index=poc_edge_index, poc_edge_attr=poc_edge_attr,
                cross_edge_index=cross_edge_index, cross_edge_attr=cross_edge_attr,
                t=t,
            )
            x = x + dt * v
            x = _clamp_update(x, x_prev, max_delta=5.0)

        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_update(x_new: Tensor, x_old: Tensor, max_delta: float = 5.0) -> Tensor:
    """
    Clip per-atom displacement to max_delta Angstroms per Euler step.
    Prevents runaway positions during early training.
    """
    delta = x_new - x_old
    norm  = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (norm / max_delta).clamp(min=1.0)  # scale >= 1 means clipping
    return x_old + delta / scale


def _extract_subgraph_edges(
    edge_index: Tensor,
    edge_attr: Tensor,
    node_mask: Tensor,
    offset: int,
) -> tuple:
    """
    Extract edges where both endpoints are in node_mask.
    Re-index to local (0-based) indices.
    """
    if edge_index.size(1) == 0:
        return edge_index, edge_attr

    src, dst = edge_index
    keep = node_mask[src] & node_mask[dst]

    if not keep.any():
        n_feat = edge_attr.size(1) if edge_attr.dim() > 1 else 0
        return (
            torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
            torch.zeros((0, n_feat), dtype=edge_attr.dtype, device=edge_attr.device),
        )

    ei_sub = edge_index[:, keep] - offset
    ea_sub = edge_attr[keep]
    return ei_sub, ea_sub


def _extract_cross_edges(
    cross_edge_index: Tensor,
    cross_edge_attr: Tensor,
    lig_mask: Tensor,
    poc_mask: Tensor,
    lig_offset: int,
    poc_offset: int,
) -> tuple:
    """
    Extract cross edges (poc→lig) for a single graph.
    Re-index both ligand and pocket indices to local space.
    """
    if cross_edge_index.size(1) == 0:
        return cross_edge_index, cross_edge_attr

    poc_idx, lig_idx = cross_edge_index
    keep = lig_mask[lig_idx] & poc_mask[poc_idx]

    if not keep.any():
        return (
            torch.zeros((2, 0), dtype=torch.long, device=cross_edge_index.device),
            torch.zeros((0, cross_edge_attr.size(1)), dtype=cross_edge_attr.dtype,
                        device=cross_edge_attr.device),
        )

    ei_sub = torch.stack([
        poc_idx[keep] - poc_offset,
        lig_idx[keep] - lig_offset,
    ], dim=0)
    ea_sub = cross_edge_attr[keep]
    return ei_sub, ea_sub
