"""
Phase 4 — EntropicDiffusionDecoder.

Wraps the VDT DiffusionDecoder, replacing its fixed cosine_alpha_bar(t, T=200)
schedule with clock.alpha_bar(tau) from the EntropicClock.

See docs/01-IMPLEMENTATION-PLAN.md §5 for the full specification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from esdm.clock import EntropicClock

# from vdeductive.diffusion_decoder import DiffusionDecoder as _VDTDiffusionDecoder


class EntropicDiffusionDecoder(nn.Module):
    """
    SDM reverse chain with tau-indexed schedule.

    The only change vs. VDT DiffusionDecoder:
        BEFORE:  alpha_bar_t = cosine_alpha_bar(t, T=200)
        AFTER:   alpha_bar_t = clock.alpha_bar(tau=clock.tau_n)
    """

    def __init__(self, dim: int, n_steps: int = 10) -> None:
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        # self._vdt_decoder = _VDTDiffusionDecoder(dim=dim, n_steps=n_steps)

    def forward(
        self,
        Q: Tensor,          # (B, F) vibrational state
        L_z: Tensor,        # (B, F, F) or (F, F) wiring Laplacian from spectral decoder
        clock: EntropicClock,  # entropic clock providing alpha_bar(tau)
    ) -> Tensor:
        """
        Returns x_hat (B, F) — reconstructed item embedding.
        """
        raise NotImplementedError("Phase 4: implement EntropicDiffusionDecoder.forward()")
