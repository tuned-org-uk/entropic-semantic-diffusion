"""
Synthetic minimal-surface point-cloud generators with analytic curvature
ground truth (mean curvature H = 0 for true minimal surfaces).

Ref: docs/001-IMPLEMENTATION-PLAN.md §5 (`data/minimal_surfaces.py`).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

SURFACES = ("enneper", "helicoid", "catenoid")


def _grid(n: int, u_range, v_range, seed: int = 3407) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(*u_range, size=n)
    v = rng.uniform(*v_range, size=n)
    return u, v


def enneper(n: int = 1024, seed: int = 3407) -> Tensor:
    """Enneper minimal surface patch. Mean curvature H = 0 analytically."""
    u, v = _grid(n, (-1.0, 1.0), (-1.0, 1.0), seed)
    x = u - u**3 / 3 + u * v**2
    y = v - v**3 / 3 + v * u**2
    z = u**2 - v**2
    pts = np.stack([x, y, z], axis=-1)
    return torch.from_numpy(pts).float()


def helicoid(n: int = 1024, pitch: float = 1.0, seed: int = 3407) -> Tensor:
    """Helicoid minimal surface. Mean curvature H = 0 analytically."""
    u, v = _grid(n, (-np.pi, np.pi), (-1.0, 1.0), seed)
    x = v * np.cos(pitch * u)
    y = v * np.sin(pitch * u)
    z = pitch * u
    pts = np.stack([x, y, z], axis=-1)
    return torch.from_numpy(pts).float()


def catenoid(n: int = 1024, c: float = 0.5, seed: int = 3407) -> Tensor:
    """Catenoid minimal surface. Mean curvature H = 0 analytically."""
    u, v = _grid(n, (-np.pi, np.pi), (c, 2.0), seed)
    x = c * np.cosh(v / c) * np.cos(u)
    y = c * np.cosh(v / c) * np.sin(u)
    z = v
    pts = np.stack([x, y, z], axis=-1)
    return torch.from_numpy(pts).float()


def sample_surface(name: str, n: int = 1024, seed: int = 3407) -> Tensor:
    if name == "enneper":
        return enneper(n, seed)
    if name == "helicoid":
        return helicoid(n, seed=seed)
    if name == "catenoid":
        return catenoid(n, seed=seed)
    raise ValueError(f"unknown surface {name!r}; choose from {SURFACES}")
