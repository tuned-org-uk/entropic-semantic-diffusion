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

        Computes the entropic time step ``delta_tau = |Δ kl_tau|``,
        the entropic pump ``Lambda_tau = 1 / |∂_φ kl_tau|``, and updates
        all persistent buffers. A vector-valued ``kl_tau_current`` is
        treated as per-mode KLs; a scalar is treated as the bright-sector
        total.
        """
        # ---- per-mode entropic time (checkpointed) ---------------------- #
        has_per_mode = "_per_mode_tau" in self._buffers
        if kl_tau_current.dim() > 0 and not has_per_mode:
            self.register_buffer("_per_mode_tau", torch.zeros_like(kl_tau_current))
        if kl_tau_current.dim() > 0:
            self._per_mode_tau = self._per_mode_tau + kl_tau_current.abs()

        # ---- scalar Δkl for tau_n and Lambda ---------------------------- #
        delta_kl = kl_tau_current - self._prev_kl_tau
        delta_tau = delta_kl.abs()
        delta_tau_scalar = delta_tau.sum() if delta_tau.dim() > 0 else delta_tau

        # Accumulate monotone entropic time
        self.tau_n = self.tau_n + delta_tau_scalar
        self.tau_max_est = torch.maximum(self.tau_max_est, self.tau_n)

        # Entropic pump Λ(τ) = 1 / |∂_φ kl_tau|  (discrete)
        d_phi = phi_current - self._prev_phi
        d_kl_scalar = delta_kl.sum() if delta_kl.dim() > 0 else delta_kl
        denom = (d_kl_scalar / (d_phi + self.eps)).abs() + self.eps
        Lambda_tau = (1.0 / denom).clamp(0.0, self.lambda_clip)

        # Heat death counter: reset on entropy flow, else increment
        if delta_tau_scalar.item() <= self.eps:
            self._flat_steps = self._flat_steps + 1
        else:
            self._flat_steps.zero_()

        # Conservation diagnostic: |kl_tau + dark_entropy - S_0|
        kl_tau_scalar = kl_tau_current.sum() if kl_tau_current.dim() > 0 else kl_tau_current
        if S_0 is not None:
            conserved = (kl_tau_scalar + dark_entropy - S_0).abs()
        else:
            conserved = torch.tensor(0.0, dtype=kl_tau_scalar.dtype, device=kl_tau_scalar.device)

        # Signed interference ratio
        rho_sign = rho_minus / (rho_plus + self.eps)

        # Persist previous-step values for next call
        self._prev_kl_tau = kl_tau_current.detach().clone()
        self._prev_phi = phi_current.detach().clone()

        # Update diagnostic buffers
        self.Lambda_tau = Lambda_tau.detach().clone()
        self.conserved = conserved.detach().clone()
        self.rho_sign = rho_sign.detach().clone()

        return ClockState(
            tau_n=float(self.tau_n),
            tau_max_est=float(self.tau_max_est),
            Lambda_tau=float(Lambda_tau),
            delta_tau=float(delta_tau_scalar),
            phi=float(phi_current),
            dark_entropy=float(dark_entropy),
            conserved=float(conserved),
            rho_sign=float(rho_sign),
        )

    # ------------------------------------------------------------------ #

    def alpha_bar(self, tau: Optional[Tensor] = None) -> Tensor:
        """
        Entropic cosine schedule:
            alpha_bar(τ) = cos²(π/2 · τ / τ_max_est)

        If ``tau`` is None, uses the current accumulated ``self.tau_n``.
        The argument is clamped to ``[0, tau_max_est]`` so the schedule
        stays numerically well-behaved.
        """
        if tau is None:
            tau = self.tau_n
        ratio = (tau / self.tau_max_est).clamp(0.0, 1.0)
        return torch.cos(ratio * (torch.pi / 2.0)).pow(2)

    # ------------------------------------------------------------------ #

    def is_heat_death(self) -> bool:
        """True if τ_n has been flat for > heat_death_patience steps."""
        return int(self._flat_steps.item()) >= self.heat_death_patience

    # ------------------------------------------------------------------ #

    @property
    def per_mode_tau(self) -> Optional[Tensor]:
        return self._buffers.get("_per_mode_tau")
