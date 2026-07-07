"""
Comparison harness — trains and samples both conditions on identical splits
and emits the metrics table for Experiment 3.

Ref: docs/002-EXPERIMENTS.md Experiment 3.
This is a skeleton: the full training loop is a sub-issue of the migration.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from esdm.data.minimal_surfaces import sample_surface
from esdm.graphs.wiring import WireGraph
from esdm.models.score3d import ScoreNet, dsm_loss
from esdm.models.score25d import ScoreNet25D, dsm_loss_25d
from esdm.sampling.euler_maruyama import euler_maruyama
from esdm.eval.metrics import chamfer_distance, rayleigh_energy_gap


@dataclass
class HarnessConfig:
    surface: str = "enneper"
    n_points: int = 1024
    k: int = 16
    seed: int = 3407
    steps_3d: int = 500
    steps_25d: int = 500
    sample_steps: int = 100


@dataclass
class HarnessResult:
    chamfer_3d: float
    chamfer_25d: float
    rayleigh_gap: float


def run(cfg: HarnessConfig) -> HarnessResult:
    torch.manual_seed(cfg.seed)
    pts = sample_surface(cfg.surface, cfg.n_points, cfg.seed)
    graph = WireGraph.build(pts, k=cfg.k, seed=cfg.seed)

    score3d = ScoreNet(dim=3)
    opt3d = torch.optim.Adam(score3d.parameters(), lr=1e-3)
    for _ in range(cfg.steps_3d):
        opt3d.zero_grad()
        loss = dsm_loss(score3d, pts)
        loss.backward()
        opt3d.step()

    z0 = graph.spectral_embedding(n_components=2)
    f0 = pts[:, :1].squeeze(-1) if pts.shape[1] >= 1 else torch.zeros(pts.shape[0])
    score25d = ScoreNet25D(item_dim=2)
    opt25d = torch.optim.Adam(score25d.parameters(), lr=1e-3)
    for _ in range(cfg.steps_25d):
        opt25d.zero_grad()
        loss = dsm_loss_25d(score25d, graph, z0, f0)
        loss.backward()
        opt25d.step()

    with torch.no_grad():
        x_gen_3d = euler_maruyama(score3d, torch.randn_like(pts), num_steps=cfg.sample_steps, seed=cfg.seed)
    cd_3d = chamfer_distance(x_gen_3d, pts).item()
    cd_25d = chamfer_distance(z0, graph.spectral_embedding(2)).item()
    r_gap = rayleigh_energy_gap(graph, f0, f0).item()
    return HarnessResult(chamfer_3d=cd_3d, chamfer_25d=cd_25d, rayleigh_gap=r_gap)
