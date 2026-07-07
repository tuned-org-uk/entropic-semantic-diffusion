"""
Regression test: the per-mode KL helper must match the vendored scalar KL
when summed and averaged. This guards against silent divergence between
esdm.models.spectral_kl and esdm._vendor.spectral.tau_mode_kl.
"""
import torch

from esdm._vendor.spectral import tau_mode_kl
from esdm.models.spectral_kl import tau_mode_kl_per_mode


def test_per_mode_matches_vendor_when_reduced():
    torch.manual_seed(3407)
    B, q = 4, 8
    log_a = torch.randn(B, q)
    log_b = torch.randn(B, q)
    eigvals_q = torch.rand(q).abs() + 0.1

    vendor_scalar = tau_mode_kl(log_a, log_b, eigvals_q, tau=1.0, a_min=0.1)
    per_mode = tau_mode_kl_per_mode(log_a, log_b, eigvals_q, tau=1.0, a_min=0.1)
    reduced = per_mode.sum(dim=-1).mean()

    assert torch.allclose(vendor_scalar, reduced, atol=1e-6)


def test_per_mode_shape():
    B, q = 3, 5
    log_a = torch.randn(B, q)
    log_b = torch.randn(B, q)
    eigvals_q = torch.rand(q).abs() + 0.1
    out = tau_mode_kl_per_mode(log_a, log_b, eigvals_q)
    assert out.shape == (B, q)
