"""
Tests for Q-space reverse sampler (issue #29 Phase 1, Step 5).
"""
from __future__ import annotations

import torch
import pytest

from esdm.models.esdm import ESDM
from esdm.sampling.entropic_sampler import entropic_reverse_sde
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


@pytest.mark.parametrize("sampler", [q_space_reverse_sde, entropic_reverse_sde])
def test_single_grid_point_preserves_chart_prior_amplitudes(sampler):
    torch.manual_seed(3407)
    model = _build_small_model().double()
    basis = model.U_q
    # With no reverse transition the Q-space path still maps Q to data;
    # the input-space path already holds data coordinates.
    z = torch.randn(4, basis.shape[1], generator=torch.Generator().manual_seed(3407),
                    dtype=basis.dtype)
    if sampler is q_space_reverse_sde:
        z = z * model.readout.weights()
    torch.testing.assert_close(sampler(model, n_samples=4, n_steps=1), z @ basis.T)


def test_q_reverse_transition_preserves_noise_and_applies_modal_gains():
    torch.manual_seed(3407)
    model = _build_small_model().double()
    # At zero wave depth the score predictor returns its actual input. Three
    # grid points isolate both the intermediate Gaussian and the terminal gain.
    model.vibrational.n_steps = 0
    basis = model.U_q
    gen = torch.Generator().manual_seed(3407)
    shape = (4, basis.shape[1])
    initial = torch.randn(shape, generator=gen, dtype=basis.dtype)
    torch.randn(shape, generator=gen, dtype=basis.dtype)  # posterior draw
    noise = torch.randn(shape, generator=gen, dtype=basis.dtype)
    gains = model.readout.weights().detach()
    a_terminal = model.clock.alpha_bar(tau=model.clock.tau_max_est).sqrt()
    a_mid = model.clock.alpha_bar(tau=model.clock.tau_max_est / 2).sqrt()
    mid = a_mid * (a_terminal * initial * gains) + (1 - a_mid.square()).sqrt() * noise
    expected = a_mid * mid * gains.square() @ basis.T
    torch.testing.assert_close(q_space_reverse_sde(model, n_samples=4, n_steps=3), expected)


def test_input_space_clean_endpoint_uses_readout_without_schedule_attenuation():
    torch.manual_seed(3407)
    model = _build_small_model().double()
    model.vibrational.n_steps = 0
    basis = model.U_q
    z = torch.randn(4, basis.shape[1], generator=torch.Generator().manual_seed(3407),
                    dtype=basis.dtype)
    torch.testing.assert_close(entropic_reverse_sde(model, n_samples=4, n_steps=2),
                               (z * model.readout.weights()) @ basis.T)


def test_q_sampler_reproducible():
    model = _build_small_model()
    s1 = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    s2 = q_space_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert torch.allclose(s1, s2, atol=1e-4)


@pytest.mark.parametrize("sampler", [q_space_reverse_sde, entropic_reverse_sde])
def test_clock_sampler_preserves_model_and_rng(sampler):
    model = _build_small_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    samples = sampler(model, n_samples=3, n_steps=4)
    assert model.training and not samples.requires_grad
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, before[k], msg=k)
