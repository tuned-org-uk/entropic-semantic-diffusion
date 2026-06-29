"""
Phase 2 — EntropicClock.

Accumulates entropic time τ_n = Σ|Δkl_tau_i| and derives:
  - delta_tau   : the entropic time step at the current batch
  - Lambda_tau  : the entropic pump  1 / |∂_φ kl_bright| + ε
  - alpha_bar   : the tau-indexed cosine diffusion schedule
  - heat_death  : True when τ_n has been flat for too long

See docs/01-IMPLEMENTATION-PLAN.md §3 for the full specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ClockState:
    tau_n: float
    tau_max_est: float
    Lambda_tau: float
    delta_tau: float
    phi: float
    dark_entropy: float
    conserved: float
    rho_sign: float


class EntropicClock(nn.Module):
    """
    Stateful module — all state lives in buffers so it is preserved across
    checkpoint save/load without contributing to any gradient.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        heat_death_patience: int = 200,
        lambda_clip: float = 100.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.heat_death_patience = heat_death_patience
        self.lambda_clip = lambda_clip

        # Scalar buffers
        self.register_buffer("tau_n",       torch.tensor(0.0))
        self.register_buffer("tau_max_est", torch.tensor(1.0))  # avoid /0 on first step
        self.register_buffer("_prev_kl_tau",torch.tensor(0.0))
        self.register_buffer("_prev_phi",   torch.tensor(0.0))
        self.register_buffer("_flat_steps", torch.tensor(0))

        # Diagnostic buffers updated by step(); health_check() reads these.
        self.register_buffer("conserved",  torch.tensor(0.0))
        self.register_buffer("rho_sign",   torch.tensor(0.0))
        self.register_buffer("Lambda_tau", torch.tensor(0.0))

        # Per-mode tau vector — initialised lazily on first step.
        # Registered as a buffer so it survives checkpoint save/load.

    # ------------------------------------------------------------------ #

    def step(
        self,
        kl_tau_current: Tensor,
        phi_current: Tensor,
        rho_plus: Tensor,
        rho_minus: Tensor,
        dark_entropy: Tensor,
        S_0: Optional[Tensor] = None,
    ) -> ClockState:
        """
        Advance the entropic clock by one step.

        Lazily creates the per-mode tau buffer the first time a vector-valued
        ``kl_tau_current`` is seen, then accumulates ``|kl_tau_current|`` into
        it. Diagnostic buffers (conserved, rho_sign, Lambda_tau) are updated
        so ``health_check()`` can read them after a checkpoint restore.
        """
        # Ensure per-mode entropic time is checkpointed. Register on first use.
        has_per_mode = "_per_mode_tau" in self._buffers
        if not has_per_mode and kl_tau_current.dim() > 0:
            self.register_buffer("_per_mode_tau", torch.zeros_like(kl_tau_current))
        elif has_per_mode:
            self._per_mode_tau = self._per_mode_tau + kl_tau_current.abs()

        # TODO: compute tau_n, tau_max_est, Lambda_tau, delta_tau, conserved, rho_sign.
        # Update the registered diagnostic buffers below as part of that work.
        raise NotImplementedError("Phase 2: implement EntropicClock.step()")

    # ------------------------------------------------------------------ #

    def alpha_bar(self, tau: Optional[Tensor] = None) -> Tensor:
        """
        Entropic cosine schedule:
            alpha_bar(τ) = cos²(π/2 · τ / τ_max_est)
        """
        raise NotImplementedError("Phase 2: implement EntropicClock.alpha_bar()")

    # ------------------------------------------------------------------ #

    def is_heat_death(self) -> bool:
        """True if τ_n has been flat for > heat_death_patience steps."""
        return int(self._flat_steps.item()) >= self.heat_death_patience

    # ------------------------------------------------------------------ #

    @property
    def per_mode_tau(self) -> Optional[Tensor]:
        return self._buffers.get("_per_mode_tau")
