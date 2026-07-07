"""
Evaluation metrics for the 2.5-D vs 3D comparison.

Ref: docs/002-EXPERIMENTS.md Experiment 3 and
docs/000-DIFFUSION-ON-PHYSICAL-SURFACES.md "Validation plan".
"""
from __future__ import annotations

import torch
from torch import Tensor

from esdm.graphs.wiring import WireGraph


def chamfer_distance(p: Tensor, q: Tensor) -> Tensor:
    """Symmetric nearest-neighbour distance between point sets p (N,3) and q (M,3)."""
    d = torch.cdist(p, q)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def rayleigh_energy_gap(graph: WireGraph, f_gen: Tensor, f_real: Tensor) -> Tensor:
    R_gen = graph.rayleigh_quotient(f_gen).mean()
    R_real = graph.rayleigh_quotient(f_real).mean()
    return (R_gen - R_real).abs()


def mean_curvature_error(points: Tensor, reference_H: float = 0.0) -> Tensor:
    """Placeholder discrete mean-curvature estimate.

    A faithful discrete mean-curvature operator on a point cloud requires
    local mesh/normal estimation (e.g. moving-least-squares). This stub
    returns the variance of nearest-neighbour distances as a cheap
    smoothness proxy and is replaced by a proper estimator in the Exp 3
    harness. Minimal surfaces have H -> 0.
    """
    nn_dist = torch.cdist(points, points)
    nn_dist.fill_diagonal_(float("inf"))
    return nn_dist.min(dim=1).values.var()
