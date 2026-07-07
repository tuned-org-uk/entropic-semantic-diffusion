"""
Phase 4 test — EntropicDiffusionDecoder.
All assertions from 01-IMPLEMENTATION-PLAN.md §5.3.
"""
import pytest
import torch

from esdm.clock import EntropicClock
from esdm.diffusion_decoder import EntropicDiffusionDecoder


def test_alpha_bar_monotone_decreasing():
    """alpha_bar decreases monotonically as tau increases."""
    clock = EntropicClock()
    clock.tau_n.fill_(0.0)
    clock.tau_max_est.fill_(10.0)

    alphas = [clock.alpha_bar(torch.tensor(t)).item() for t in range(11)]
    for i in range(1, len(alphas)):
        assert alphas[i] <= alphas[i - 1] + 1e-6


def test_alpha_bar_bounds():
    """alpha_bar is ~1 at tau=0 and ~0 at tau=tau_max."""
    clock = EntropicClock()
    clock.tau_max_est.fill_(100.0)

    assert abs(float(clock.alpha_bar(torch.tensor(0.0))) - 1.0) < 1e-5
    assert abs(float(clock.alpha_bar(torch.tensor(100.0))) - 0.0) < 1e-5


def test_per_mode_alpha_bar_independent():
    """Per-mode alpha_bar values depend only on their own tau."""
    clock = EntropicClock()
    clock.tau_max_est.fill_(10.0)
    clock.register_buffer("_per_mode_tau", torch.tensor([0.0, 5.0, 10.0]))

    decoder = EntropicDiffusionDecoder(dim=3, use_per_mode=True)
    Q = torch.randn(2, 3)
    x_hat = decoder(Q, torch.eye(3), clock, noise=torch.zeros_like(Q))

    # With zero noise, x_hat = sqrt(alpha_bar) * Q.  The three columns should
    # be scaled by three different alpha_bar values.
    scales = x_hat[0] / (Q[0] + 1e-12)
    assert scales[0] > scales[1] > scales[2] - 1e-6


def test_diffusion_decoder_output_shape_and_finite():
    """Full reverse-chain pass produces finite x_hat with correct shape."""
    clock = EntropicClock()
    clock.tau_n.fill_(2.5)
    clock.tau_max_est.fill_(10.0)

    decoder = EntropicDiffusionDecoder(dim=16)
    Q = torch.randn(4, 16)
    L_z = torch.eye(16)

    x_hat = decoder(Q, L_z, clock)
    assert x_hat.shape == Q.shape
    assert torch.all(torch.isfinite(x_hat))


def test_diffusion_decoder_clean_start_full_noise():
    """At tau=0 the output equals Q; at tau=tau_max it equals noise."""
    clock = EntropicClock()
    clock.tau_max_est.fill_(10.0)
    decoder = EntropicDiffusionDecoder(dim=8)
    Q = torch.randn(2, 8)
    noise = torch.randn_like(Q)

    clock.tau_n.fill_(0.0)
    x_hat_start = decoder(Q, torch.eye(8), clock, noise=noise)
    assert torch.allclose(x_hat_start, Q, atol=1e-5)

    clock.tau_n.fill_(10.0)
    x_hat_end = decoder(Q, torch.eye(8), clock, noise=noise)
    assert torch.allclose(x_hat_end, noise, atol=1e-5)
