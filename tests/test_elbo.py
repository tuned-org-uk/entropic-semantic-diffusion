"""
Phase 5 test — esdm_elbo.
All assertions from 01-IMPLEMENTATION-PLAN.md §6.3.
"""
import torch
from esdm.elbo import esdm_elbo


def _make_tensors(B: int = 8, F: int = 32):
    torch.manual_seed(0)
    x     = torch.randn(B, F)
    x_hat = torch.randn(B, F)
    kl_z  = torch.tensor(0.5)
    kl_S  = torch.tensor(0.1)
    kl_tau = torch.tensor(0.3)
    dark   = torch.tensor(0.2)
    S_0    = (kl_tau + dark).detach().clone()
    return x, x_hat, kl_z, kl_S, kl_tau, dark, S_0


def test_elbo_returns_finite():
    x, x_hat, kl_z, kl_S, kl_tau, dark, S_0 = _make_tensors()
    loss, metrics = esdm_elbo(x, x_hat, kl_z, kl_S, kl_tau, dark, S_0)
    assert torch.isfinite(loss)


def test_conserved_zero_when_exact():
    x, x_hat, kl_z, kl_S, kl_tau, dark, S_0 = _make_tensors()
    # S_0 = kl_tau + dark exactly → conserved should be ~0
    _, metrics = esdm_elbo(x, x_hat, kl_z, kl_S, kl_tau, dark, S_0)
    assert float(metrics["conserved"]) < 1e-6


def test_S0_no_gradient():
    x, x_hat, kl_z, kl_S, kl_tau, dark, S_0 = _make_tensors()
    # Give at least one tensor a gradient so the loss has a grad_fn.
    x_hat = x_hat.requires_grad_(True)
    S_0.requires_grad_(False)
    loss, _ = esdm_elbo(x, x_hat, kl_z, kl_S, kl_tau, dark, S_0)
    loss.backward()
    assert x_hat.grad is not None
    assert S_0.grad is None


def test_all_metric_keys_present():
    x, x_hat, kl_z, kl_S, kl_tau, dark, S_0 = _make_tensors()
    _, metrics = esdm_elbo(x, x_hat, kl_z, kl_S, kl_tau, dark, S_0)
    for key in ("recon", "kl_z", "kl_S", "kl_tau", "dark_entropy", "conserved", "beta"):
        assert key in metrics, f"missing metric key: {key}"
