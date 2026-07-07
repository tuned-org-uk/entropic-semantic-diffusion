"""
3D volumetric score network + denoising score-matching loss (control condition).

Ref: docs/000-DIFFUSION-ON-PHYSICAL-SURFACES.md "Update equations" and
docs/001-IMPLEMENTATION-PLAN.md §4.1, §5 (`models/score3d.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ScoreNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1))


def sigma_fn(t: Tensor, sigma_min: float = 0.01, sigma_max: float = 1.0) -> Tensor:
    return sigma_min * (sigma_max / sigma_min) ** t


def dsm_loss(score_net: nn.Module, x0: Tensor, T: float = 1.0) -> Tensor:
    b = x0.shape[0]
    t = torch.rand(b, 1, device=x0.device) * T
    sigma = sigma_fn(t).to(x0.dtype)
    noise = torch.randn_like(x0)
    xt = x0 + sigma * noise
    target = -noise / (sigma + 1e-8)
    pred = score_net(xt, t)
    return ((pred - target) ** 2).sum(-1).mean()
