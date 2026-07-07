"""Smoke tests for the new 2.5-D dual-space track (esdm.*)."""
import torch

from esdm.data.minimal_surfaces import sample_surface, SURFACES
from esdm.graphs.wiring import WireGraph
from esdm.models.score3d import ScoreNet, dsm_loss
from esdm.models.score25d import ScoreNet25D, dsm_loss_25d, graph_diffusion_step
from esdm.sampling.euler_maruyama import euler_maruyama
from esdm.eval.harness import run, HarnessConfig


def test_surfaces_present():
    assert SURFACES == ("enneper", "helicoid", "catenoid")


def test_surface_shape():
    pts = sample_surface("enneper", 64, seed=3407)
    assert pts.shape == (64, 3) and torch.isfinite(pts).all()


def test_wiregraph_contract():
    pts = sample_surface("catenoid", 64, seed=3407)
    g = WireGraph.build(pts, k=8, seed=3407)
    L = g.laplacian(normalized=True)
    assert L.is_sparse and L.shape == (64, 64)
    R = g.rayleigh_quotient(pts[:, 0])
    assert R.shape[0] == 64 and torch.isfinite(R).all()
    z = g.spectral_embedding(2)
    assert z.shape == (64, 2)


def test_3d_loss_finite():
    pts = sample_surface("helicoid", 64, seed=3407)
    loss = dsm_loss(ScoreNet(3), pts)
    assert torch.isfinite(loss) and loss > 0


def test_25d_loss_finite():
    pts = sample_surface("enneper", 64, seed=3407)
    g = WireGraph.build(pts, k=8, seed=3407)
    z = g.spectral_embedding(2).detach()
    loss = dsm_loss_25d(ScoreNet25D(2), g, z, pts[:, 0].detach())
    assert torch.isfinite(loss) and loss > 0


def test_graph_diffusion_step_smooths():
    pts = sample_surface("enneper", 64, seed=3407)
    g = WireGraph.build(pts, k=8, seed=3407)
    L = g.laplacian(normalized=True)
    f = torch.randn(64)
    R_before = g.rayleigh_quotient(f).abs().mean()
    f2 = graph_diffusion_step(f, L, sigma=torch.tensor(1.0), dt=torch.tensor(0.1))
    R_after = g.rayleigh_quotient(f2).abs().mean()
    assert R_after <= R_before + 1e-3


def test_sampler_shape():
    out = euler_maruyama(ScoreNet(3), torch.randn(8, 3), num_steps=4, seed=3407)
    assert out.shape == (8, 3)


def test_harness_runs():
    r = run(HarnessConfig(n_points=32, steps_3d=1, steps_25d=1, sample_steps=2))
    assert isinstance(r.chamfer_3d, float) and isinstance(r.chamfer_25d, float)
