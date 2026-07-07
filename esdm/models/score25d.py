"""
2.5-D dual-space score network + graph-diffusion step (treatment condition).

State is split into a 2D item coordinate z_t and a feature signal f_t on the
wired graph. Update equations (docs/001-IMPLEMENTATION-PLAN.md §4.2):

    z_{t-dt} = z_t + eta * grad_z R(f_t) * dt + sigma * sqrt(dt) * eps_z
    f_{t-dt} = f_t - (sigma^2 * dt / 2) * L_F f_t
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from esdm.graphs.wiring import WireGraph


class ScoreNet25D(nn.Module):
    def __init__(self, item_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.item_dim = item_dim
        # f_t is injected as a conditioning feature (one scalar per item: its Rayleigh score)
        self.net = nn.Sequential(
            nn.Linear(item_dim + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, item_dim),
        )

    def forward(self, z: Tensor, t: Tensor, f_score: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        if f_score.dim() == 1:
            f_score = f_score.unsqueeze(-1)
        return self.net(torch.cat([z, t, f_score], dim=-1))


def graph_diffusion_step(f: Tensor, L: Tensor, sigma: Tensor, dt: Tensor) -> Tensor:
    """f_{t-dt} = f_t - (sigma^2 dt / 2) L f_t  (sparse Laplacian-vector product)."""
    if L.is_sparse:
        Lf = torch.sparse.mm(L, f.unsqueeze(-1)).squeeze(-1)
    else:
        Lf = L @ f
    return f - 0.5 * (sigma ** 2) * dt * Lf


def dsm_loss_25d(
    score_net: nn.Module, graph: WireGraph, z0: Tensor, f0: Tensor, T: float = 1.0
) -> Tensor:
    b = z0.shape[0]
    t = torch.rand(b, 1, device=z0.device) * T
    sigma = (0.01 * (100.0) ** t).to(z0.dtype)
    noise = torch.randn_like(z0)
    zt = z0 + sigma * noise
    target = -noise / (sigma + 1e-8)
    f_score = graph.rayleigh_quotient(f0).detach()
    if f_score.dim() == 0:
        f_score = f_score.expand(b, 1)
    elif f_score.shape[0] != b:
        f_score = f_score[:b]
    if f_score.dim() == 1:
        f_score = f_score.unsqueeze(-1)
    pred = score_net(zt, t, f_score)
    return ((pred - target) ** 2).sum(-1).mean()
