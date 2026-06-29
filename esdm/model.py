"""
Phase 6 — Top-level ESDM nn.Module.

Assembles EntropicClock, VibrationalStateBlock, SpectralLoadingDecoder,
EntropicDiffusionDecoder, and the ELBO into one end-to-end model.

See docs/01-IMPLEMENTATION-PLAN.md §7 for the full specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from esdm.clock import EntropicClock, ClockState
from esdm.elbo  import esdm_elbo


@dataclass
class ESDMOutput:
    x_hat:   Tensor
    loss:    Tensor
    clock:   ClockState
    metrics: dict[str, Tensor]


class ESDM(nn.Module):
    """
    Entropic Semantic Diffusion Model.

    Sub-modules are instantiated lazily via _build(); call build() explicitly
    before the first forward pass once you have the config.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.clock = EntropicClock(
            eps=cfg["clock"]["eps"],
            heat_death_patience=cfg["clock"]["heat_death_patience"],
            lambda_clip=cfg["clock"]["lambda_clip"],
        )
        # S_0 registered as a buffer so it is saved in checkpoints
        self.register_buffer("S_0", None)  # set on first forward pass
        self.register_buffer("_last_N_active", torch.tensor(0))

        # Sub-module stubs — uncomment as phases are implemented
        # from esdm.laplacian           import DualSpaceMatrix
        # from esdm.vibrational         import VibrationalStateBlock
        # from esdm.spectral_decoder    import SpectralLoadingDecoder
        # from esdm.diffusion_decoder   import EntropicDiffusionDecoder
        # self.dual_space        = DualSpaceMatrix(**cfg["fusion"])
        # self.vibrational       = VibrationalStateBlock(...)
        # self.spectral_decoder  = SpectralLoadingDecoder(...)
        # self.diffusion_decoder = EntropicDiffusionDecoder(...)

    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor) -> ESDMOutput:
        raise NotImplementedError(
            "ESDM.forward(): implement after all Phase 1-5 modules are complete."
        )

    # ------------------------------------------------------------------ #

    def health_check(self) -> dict:
        """Diagnostic summary — call every N batches, not every step."""
        c = self.clock
        return {
            "tau_n":      float(c.tau_n),
            "tau_flat":   int(c._flat_steps),
            "conserved":  float(getattr(c, "conserved", float("nan"))),
            "rho_sign":   float(getattr(c, "rho_sign",  float("nan"))),
            "N_active":   int(self._last_N_active),
            "Lambda_tau": float(getattr(c, "Lambda_tau", float("nan"))),
            "heat_death": c.is_heat_death(),
        }
