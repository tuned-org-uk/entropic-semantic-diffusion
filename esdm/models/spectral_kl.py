"""
Per-mode spectral KL helper for the ESDM model.

De-duplicates the Gamma-vs-Exponential mode KL that was previously inlined
in ``ESDM._tau_mode_kl_per_mode``. The vendored ``tau_mode_kl`` in
``esdm._vendor.spectral`` returns a reduced scalar (sum over modes, mean
over batch); this module exposes the unreduced ``(B, q)`` tensor so the
ESDM model can split modes into bright/active and dark/inactive sectors
before reduction.

The formula is identical to ``esdm._vendor.spectral.tau_mode_kl`` -- kept in
sync by the regression test ``test_spectral_kl_matches_vendor``.
"""
from __future__ import annotations

import torch
from torch import Tensor


def tau_mode_kl_per_mode(
    log_a: Tensor,
    log_b: Tensor,
    eigvals_q: Tensor,
    tau: float = 1.0,
    a_min: float = 0.1,
) -> Tensor:
    """
    Per-mode KL( Gamma(a,b) || Exp(tau * lambda_k) ), shape (B, q).

    Returns the unreduced KL tensor so the caller can split modes into
    bright (active) and dark (inactive) sectors before summation.

    Parameters mirror ``esdm._vendor.spectral.tau_mode_kl``.
    """
    a = log_a.exp().clamp(min=a_min)  # (B, q)
    b = log_b.exp()                   # (B, q)
    r = (tau * eigvals_q.clamp(min=1e-6)).unsqueeze(0)  # (1, q)

    return (
        log_b
        - r.log()
        - torch.lgamma(a)
        - (1.0 - a) * torch.digamma(a)
        - a
        + a * r / b
    )  # (B, q)
