"""
Reverse-SDE Euler–Maruyama sampler, parameterised by dimensionality.

Used for both the 3D baseline (dim=3) and, with the graph-diffusion step
injected, the 2.5-D dual-space model.

Ref: docs/000-DIFFUSION-ON-PHYSICAL-SURFACES.md "Sampling" and
docs/001-IMPLEMENTATION-PLAN.md §5 (`sampling/euler_maruyama.py`).
"""
from __future__ import annotations

import torch
from torch import Tensor

from esdm.models.score3d import sigma_fn


def euler_maruyama(
    score_net, x_T: Tensor, num_steps: int = 100, T: float = 1.0,
    device=None, seed: int = 3407,
) -> Tensor:
    gen = torch.Generator(device=x_T.device).manual_seed(seed)
    dt = T / num_steps
    x = x_T
    for i in range(num_steps):
        t_val = T - i * dt
        t = torch.full((x.shape[0], 1), t_val, device=x_T.device, dtype=x_T.dtype)
        sigma = sigma_fn(t).to(x_T.dtype)
        score = score_net(x, t)
        x = x + 0.5 * (sigma ** 2) * score * dt + sigma * (dt ** 0.5) * torch.randn(x.shape, generator=gen, device=x_T.device, dtype=x_T.dtype)
    return x
