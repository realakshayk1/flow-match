"""
SE(3)-Equivariant Graph Neural Network (EGNN) implementation from scratch.
Based on: Satorras et al., 2021 — "E(n) Equivariant Graph Neural Networks"

No e3nn dependency. Designed for CPU inference with < 500K parameters.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# EGNNLayer
# ---------------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """
    Single EGNN message-passing layer.

    Operates on a combined node set [ligand atoms || pocket atoms].
    Ligand atom coordinates are updated; pocket coordinates are held fixed
    via the fixed_mask argument.

    Args:
        hidden_dim: node feature dimension (same for input and output)
        edge_dim:   edge attribute dimension

    Forward:
        h          [N_total, hidden_dim]
        x          [N_total, 3]
        edge_index [2, E_total]    — row=source, col=target (messages flow src→dst)
        edge_attr  [E_total, edge_dim]
        fixed_mask [N_total] bool  — True = pocket atom, coordinates not updated

    Returns:
        h_new [N_total, hidden_dim]
        x_new [N_total, 3]
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        act = nn.SiLU()

        # Message network: [h_i, h_j, ||x_i - x_j||², edge_attr] → message
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim), act,
        )

        # Coordinate update weight: message → scalar weight (no final activation)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act,
            nn.Linear(hidden_dim, 1),
        )

        # Node update: [h_i, aggregated_message] → h_new
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        fixed_mask: Optional[Tensor] = None,
    ):
        src, dst = edge_index  # messages flow src → dst; aggregate at dst

        diff    = x[src] - x[dst]                          # [E, 3] — rotation equivariant
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)  # [E, 1] — rotation invariant

        # Normalise distance so phi_e receives values in [0, ~1] regardless of
        # absolute coordinate scale.  Dividing by (10 Å)² keeps typical protein-
        # ligand distances (1-20 Å → dist_sq 1-400 Å²) in the range [0.01, 4].
        # Without this, random weights × raw dist_sq (up to 400) blow up the
        # messages and cause coordinate explosion → inf/NaN loss before step 1.
        dist_sq_norm = dist_sq / 100.0                     # [E, 1]

        # Compute messages
        m_ij = self.phi_e(
            torch.cat([h[src], h[dst], dist_sq_norm, edge_attr], dim=-1)
        )  # [E, hidden_dim]

        # Equivariant coordinate update: tanh bounds each edge's contribution to
        # [-1, 1] × ||diff||, preventing runaway coordinate growth across layers.
        coord_weight = self.phi_x(m_ij).tanh()             # [E, 1]  ∈ (-1, 1)
        coord_delta  = diff * coord_weight                  # [E, 3] — equivariant

        coord_agg = torch.zeros_like(x)                # [N, 3]
        coord_agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(coord_delta),
            coord_delta,
        )

        # Mask out pocket atom coordinate updates
        if fixed_mask is not None:
            update_mask = (~fixed_mask).float().unsqueeze(-1)  # [N, 1]
            coord_agg = coord_agg * update_mask

        # Node feature update via aggregated messages (with residual)
        msg_agg = torch.zeros(h.size(0), m_ij.size(-1), device=h.device, dtype=h.dtype)
        msg_agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(m_ij),
            m_ij,
        )

        h_new = h + self.phi_h(torch.cat([h, msg_agg], dim=-1))  # residual
        x_new = x + coord_agg

        return h_new, x_new


# ---------------------------------------------------------------------------
# EGNNFlowModel
# ---------------------------------------------------------------------------

class EGNNFlowModel(nn.Module):
    """
    Full EGNN velocity field for flow matching.

    Takes a ligand graph and a pocket graph, builds a combined graph
    with pocket→ligand cross edges, runs 4 EGNN layers, and outputs
    per-ligand-atom velocity vectors.

    Design:
    - Pocket coordinates are never updated (fixed_mask)
    - Cross edges carry messages only from pocket → ligand
    - Time t is embedded into ligand node features (not pocket)
    - All edge types are projected to the same hidden_dim before layers

    Args:
        lig_node_dim:   int — 17 (ligand atom features)
        poc_node_dim:   int — 29 (pocket atom features)
        lig_edge_dim:   int — 6  (ligand bond features)
        poc_edge_dim:   int — 16 (RBF-encoded pocket edges)
        cross_edge_dim: int — 16 (RBF-encoded cross edges)
        hidden_dim:     int — 64
        n_layers:       int — 4

    Forward:
        lig_x            [N_lig, 3]       noisy ligand coords at time t
        lig_h            [N_lig, 17]      ligand node features (static)
        poc_x            [M_poc, 3]       pocket coords (fixed throughout)
        poc_h            [M_poc, 29]      pocket node features (static)
        lig_edge_index   [2, E_lig]
        lig_edge_attr    [E_lig, 6]
        poc_edge_index   [2, E_poc]       pocket-local indices
        poc_edge_attr    [E_poc, 16]
        cross_edge_index [2, N_lig*k]     (poc_local_idx, lig_local_idx)
        cross_edge_attr  [N_lig*k, 16]
        t                scalar or [1]    time in [0, 1]

    Returns:
        v [N_lig, 3] — predicted velocity (equivariant under SE(3))
    """

    def __init__(
        self,
        lig_node_dim: int = 17,
        poc_node_dim: int = 29,
        lig_edge_dim: int = 6,
        poc_edge_dim: int = 16,
        cross_edge_dim: int = 16,
        hidden_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input embedding: ligand nodes get time concatenated (+1)
        self.lig_emb = nn.Linear(lig_node_dim + 1, hidden_dim)
        self.poc_emb = nn.Linear(poc_node_dim, hidden_dim)

        # Project all edge types to the same hidden_dim
        self.lig_edge_emb = nn.Linear(lig_edge_dim, hidden_dim)
        self.poc_edge_emb = nn.Linear(poc_edge_dim, hidden_dim)
        self.cross_edge_emb = nn.Linear(cross_edge_dim, hidden_dim)

        # EGNN layers — all operate on edge_dim = hidden_dim
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output scale: invariant scalar per atom that weights the
        # equivariant coordinate displacement → velocity.
        # v = out_scale(h) * (x_updated - x_input)
        # This keeps the output equivariant: h is invariant, so scale is
        # invariant; (x_updated - x_input) is equivariant; product is equivariant.
        self.out_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        lig_x: Tensor,
        lig_h: Tensor,
        poc_x: Tensor,
        poc_h: Tensor,
        lig_edge_index: Tensor,
        lig_edge_attr: Tensor,
        poc_edge_index: Tensor,
        poc_edge_attr: Tensor,
        cross_edge_index: Tensor,
        cross_edge_attr: Tensor,
        t: Tensor,
    ) -> Tensor:
        N_lig = lig_h.size(0)
        M_poc = poc_h.size(0)

        # Embed time into ligand features.
        # t may be scalar, [1], [N_lig] (per-atom for batched training), or [N_lig, 1].
        if t.dim() == 0 or t.numel() == 1:
            t_feat = t.reshape(1, 1).expand(N_lig, 1)  # [N_lig, 1]
        else:
            t_feat = t.reshape(-1, 1)  # [N_lig, 1]

        h_lig = self.lig_emb(torch.cat([lig_h, t_feat], dim=-1))  # [N_lig, D]
        h_poc = self.poc_emb(poc_h)                                  # [M_poc, D]

        # Combine into a single node tensor [N_lig + M_poc, D]
        h = torch.cat([h_lig, h_poc], dim=0)
        x = torch.cat([lig_x, poc_x], dim=0)
        x_lig_input = lig_x.clone()  # save only ligand input coords — velocity = x_updated - x_input

        # Build combined edge index — shift pocket indices to combined space
        # Ligand edges: indices already in [0, N_lig)
        # Pocket edges: shift by N_lig to [N_lig, N_lig + M_poc)
        poc_ei_shifted = poc_edge_index + N_lig

        # Cross edges: pocket source shifted, ligand target stays as-is
        cross_ei_shifted = torch.stack([
            cross_edge_index[0] + N_lig,  # pocket source → combined space
            cross_edge_index[1],           # ligand target → combined space (already correct)
        ], dim=0)

        # Concatenate all edges
        combined_ei = torch.cat([lig_edge_index, poc_ei_shifted, cross_ei_shifted], dim=1)

        # Embed and concatenate all edge attributes
        combined_attr = torch.cat([
            self.lig_edge_emb(lig_edge_attr),
            self.poc_edge_emb(poc_edge_attr) if poc_edge_attr.size(0) > 0
                else torch.zeros(0, self.hidden_dim, device=h.device),
            self.cross_edge_emb(cross_edge_attr),
        ], dim=0)

        # Fixed mask: pocket atoms do not update coordinates
        fixed_mask = torch.zeros(N_lig + M_poc, dtype=torch.bool, device=h.device)
        fixed_mask[N_lig:] = True

        # Run EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, combined_ei, combined_attr, fixed_mask)

        # Equivariant velocity output:
        #   coord_displacement = x_updated - x_input  → equivariant (rotates with input)
        #   scale              = out_scale(h)          → invariant scalar
        #   v                  = scale * displacement  → equivariant
        coord_displacement = x[:N_lig] - x_lig_input        # [N_lig, 3]
        scale = self.out_scale(h[:N_lig])                   # [N_lig, 1]
        v = scale * coord_displacement                       # [N_lig, 3]
        return v


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_default_model(**kwargs) -> EGNNFlowModel:
    """Build model with plan-specified defaults."""
    defaults = dict(
        lig_node_dim=17,
        poc_node_dim=29,
        lig_edge_dim=6,
        poc_edge_dim=16,
        cross_edge_dim=16,
        hidden_dim=64,
        n_layers=4,
    )
    defaults.update(kwargs)
    return EGNNFlowModel(**defaults)
