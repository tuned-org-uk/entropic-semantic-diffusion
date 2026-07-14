"""
Phase 6 test — ESDM top-level model.
Smoke-tests only until all sub-modules are implemented.
"""
import numpy as np
import yaml
from pathlib import Path


def _default_cfg():
    p = Path("esdm/_configs/default.yaml")
    with p.open() as f:
        return yaml.safe_load(f)


def test_esdm_instantiates():
    from esdm.models.esdm import ESDM
    model = ESDM(_default_cfg())
    assert model is not None


def test_health_check_runs():
    from esdm.models.esdm import ESDM
    model = ESDM(_default_cfg())
    hc = model.health_check()
    assert "tau_n" in hc
    assert "heat_death" in hc


def test_forward_pass():
    import torch
    from esdm.models.esdm import ESDM
    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    model = ESDM(cfg)
    x = torch.randn(4, cfg["model"]["dim_input"])
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert torch.isfinite(out.loss)
    assert out.clock is not None




def test_forward_pass_with_wht_filter():
    """Forward pass with a timestep-aware Graph-WHT filter remains stable."""
    import torch
    from esdm.models.esdm import ESDM
    from esdm.graphs.wht_filter import GraphWHTFilter

    cfg = _default_cfg()
    F = 32
    cfg["model"]["dim_input"] = F
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = F
    cfg["model"]["n_modes"] = 8

    # Small symmetric PSD feature-space Laplacian for the WHT filter.
    rng = np.random.default_rng(3407)
    A = rng.standard_normal((F, F))
    L_F = A.T @ A
    np.fill_diagonal(L_F, L_F.diagonal() + 1e-3)
    L_F = (L_F + L_F.T) / 2.0

    cfg["model"]["wht_filter"] = GraphWHTFilter(L_matrix=L_F).fit()

    model = ESDM(cfg)
    x = torch.randn(4, F)
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert torch.isfinite(out.loss)
    assert out.clock is not None


def test_training_loop_advances_entropic_time():
    """A short training loop must increase tau_n and decrease alpha_bar."""
    import torch
    from esdm.models.esdm import ESDM

    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    cfg["model"]["n_vibrational_steps"] = 2

    torch.manual_seed(42)
    model = ESDM(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    taus = []
    alphas = []
    for step in range(100):
        x = torch.randn(8, cfg["model"]["dim_input"])
        optimizer.zero_grad()
        out = model(x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        taus.append(out.clock.tau_n)
        alphas.append(float(model.clock.alpha_bar()))

    # Entropic time must be strictly increasing across the loop.
    assert all(taus[i + 1] > taus[i] for i in range(len(taus) - 1))
    # Diffusion schedule must decrease as tau accumulates.
    assert alphas[-1] < alphas[0]
    # Loss stayed finite.
    assert all(torch.isfinite(torch.tensor(a)) for a in alphas)
