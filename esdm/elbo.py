"""
Phase 5 — ESDM ELBO with entropy conservation term.

L_ESDM = recon_loss + kl_z + kl_S + kl_tau + β · |kl_bright + kl_dark - S_0|²

where β = 1/S_0 by default (dimensionless).
S_0 is set from the first forward pass and never updated.

All terms are positive and add to the total cost (lower-is-better loss convention).
This matches the standard variational bound written as a loss to be minimised.

See docs/01-IMPLEMENTATION-PLAN.md §6 for the full specification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_log_likelihood(x: Tensor, x_hat: Tensor) -> Tensor:
    """Per-sample Gaussian log-likelihood (mean over batch and feature dims)."""
    return -F.mse_loss(x_hat, x, reduction="mean")


def esdm_elbo(
    x: Tensor,
    x_hat: Tensor,
    kl_z: Tensor,
    kl_S: Tensor,
    kl_tau: Tensor,
    dark_entropy: Tensor,
    S_0: Tensor,
    beta: Optional[float] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Compute the ESDM ELBO and return (loss, metrics_dict).

    All terms are returned as positive scalars that sum to the total loss
    (lower is better). KL terms are added to the reconstruction cost, which
    is the loss-minimisation equivalent of subtracting them from the ELBO.

    Parameters
    ----------
    x            : (B, F) input
    x_hat        : (B, F) reconstruction
    kl_z         : KL(q(z) || N(0,I))
    kl_S         : KL(q(S) || p(S|I))  — coupling / barrier weight
    kl_tau       : KL(q(ω) || p(ω|τ,L)) — bright-sector entropy
    dark_entropy : KL sum over inactive modes  — dark-sector entropy
    S_0          : initial total entropy (buffer, not parameter)
    beta         : conservation weight; None → 1/S_0
    """
    recon_loss = -gaussian_log_likelihood(x, x_hat)
    conserved = (kl_S + kl_tau + dark_entropy - S_0).abs()
    if beta is None:
        _beta: Tensor = 1.0 / (S_0.detach() + 1e-8)
    else:
        _beta = torch.tensor(beta, dtype=x.dtype, device=x.device)

    loss = recon_loss + kl_z + kl_S + kl_tau + _beta * conserved.pow(2)
    return loss, {
        "recon":        recon_loss.detach(),
        "kl_z":         kl_z.detach(),
        "kl_S":         kl_S.detach(),
        "kl_tau":       kl_tau.detach(),
        "dark_entropy": dark_entropy.detach(),
        "conserved":    conserved.detach(),
        "beta":         _beta.detach() if isinstance(_beta, Tensor) else torch.tensor(_beta),
    }
