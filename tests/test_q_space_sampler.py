"""
Tests for Q-space reverse sampler (issue #29 Phase 1, Step 5).
"""
from __future__ import annotations

import torch

from esdm.models.esdm import ESDM
from esdm.sampling.q_space_sampler import q_space_reverse_sde


def _build_small_model():
    F = 16
    cfg = {
        "model": {
            "dim_input": F, "dim_latent": F // 4, "n_nodes": F,
            "n_modes": F // 4, "n_vibrational_steps": 2, "gamma": 0.1,
            "lam_s": 1.0, "sigma_sq": 1.0, "freeze_h": False,
        },
        "clock": {"eps": 1e-8, "heat_death_patience": 200, "eps_rel": 0.0},
        "training": {"batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "max_steps": 10},
    }
    A = torch.rand(F, F)
    A = (A + A.T) / 2
    A.fill_diagonal_(0)
    D = A.sum(dim=-1).diag()
    L = D - A
    cfg["model"]["L_base"] = L
    model = ESDM(cfg)
    x = torch.randn(4, F)
    with torch.no_grad():
        _ = model(x)
    return model


def test_q_sampler_output_shape():
    model = _build_small_model()
    samples = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert samples.shape == (4, 16)
    assert torch.all(torch.isfinite(samples))


def test_q_sampler_normalized():
    model = _build_small_model()
    samples = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    norms = samples.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_q_sampler_reproducible():
    model = _build_small_model()
    s1 = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    s2 = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert torch.allclose(s1, s2, atol=1e-4)


def test_q_sampler_restores_state():
    model = _build_small_model()
    tau_before = float(model.clock.tau_n.item())
    qtm1_before = model.vibrational.Q_tm1.clone()
    _ = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert float(model.clock.tau_n.item()) == tau_before
    assert torch.equal(model.vibrational.Q_tm1, qtm1_before)
