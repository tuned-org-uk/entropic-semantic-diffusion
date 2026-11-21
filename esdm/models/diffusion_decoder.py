"""
Phase 4 — EntropicDiffusionDecoder.

Implements the ESDM reverse chain directly from the design formula, replacing
the fixed SDM ``cosine_alpha_bar(t, T=200)`` schedule with the entropic-time
schedule provided by ``EntropicClock``:

    alpha_bar_t = clock.alpha_bar(tau=clock.tau_n)
    x_hat = sqrt(alpha_bar_t) * signal + sqrt(1 - alpha_bar_t) * noise

In the clock model, ``signal`` is the bounded spectral readout of the raw
wave. The same gains never multiply the independently supplied noise.

The VDT ``DiffusionDecoder`` is not used as a black box here because its
interface is built around an embedding table ``E: (N, D)`` and node indices,
which have no natural ESDM counterpart: ``Q: (B, F)`` is already the
embedding to be reconstructed.  We do import the VDT decoder as a utility
source for ``recon_loss()`` and ``TauModeDiffusion`` if future extensions
need them.

See docs/01-IMPLEMENTATION-PLAN.md §5 for the full specification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .clock import EntropicClock


class EntropicDiffusionDecoder(nn.Module):
    """
    SDM reverse chain with tau-indexed schedule.

    Parameters
    ----------
    dim : int
        Feature dimension ``F`` of the vibrational state ``Q`` and of the
        reconstructed output ``x_hat``.
    use_per_mode : bool
        If ``True`` and ``clock.per_mode_tau`` is available, compute a
        separate ``alpha_bar`` for each feature mode.  This lets different
        modes reach their noise floor at different entropic rates.
    """

    def __init__(self, dim: int, use_per_mode: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.use_per_mode = use_per_mode

    def forward(
        self,
        Q: Tensor,              # (B, F) signal after spectral readout
        L_z: Tensor,            # (B, F, F) or (F, F) wiring Laplacian (unused here)
        clock: EntropicClock,   # entropic clock providing alpha_bar(tau)
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns x_hat (B, F).

        ``noise`` defaults to standard Gaussian if not provided, matching the
        diffusion prior.  Pass ``noise=torch.zeros_like(Q)`` for a fully
        deterministic reconstruction.
        """
        if Q.dim() != 2:
            raise ValueError(f"EntropicDiffusionDecoder expects Q of shape (B, F). Got {tuple(Q.shape)}.")

        if self.use_per_mode and clock.per_mode_tau is not None:
            alpha_bar = clock.alpha_bar(tau=clock.per_mode_tau)  # (F,)
        else:
            alpha_bar = clock.alpha_bar()  # scalar

        alpha_bar = alpha_bar.to(dtype=Q.dtype, device=Q.device)

        if noise is None:
            noise = torch.randn_like(Q)

        a_sqrt = alpha_bar.sqrt()
        noise_sqrt = (1.0 - alpha_bar).sqrt()
        return a_sqrt * Q + noise_sqrt * noise
