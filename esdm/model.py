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
from esdm.diffusion_decoder import EntropicDiffusionDecoder
from esdm.elbo  import esdm_elbo
from esdm.spectral_decoder import SpectralLoadingDecoder
from esdm.vibrational import VibrationalStateBlock


@dataclass
class ESDMOutput:
    x_hat:   Tensor
    loss:    Tensor
    clock:   ClockState
    metrics: dict[str, Tensor]


class ESDM(nn.Module):
    """
    Entropic Semantic Diffusion Model.

    S_0 is initialised from the first forward pass as the total sector
    entropy (kl_S + kl_tau + dark_entropy) and is then frozen as a buffer.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim_input = cfg["model"]["dim_input"]
        self.dim_latent = cfg["model"]["dim_latent"]
        self.n_modes = cfg["model"]["n_modes"]

        self.clock = EntropicClock(
            eps=cfg["clock"]["eps"],
            heat_death_patience=cfg["clock"]["heat_death_patience"],
            lambda_clip=cfg["clock"]["lambda_clip"],
        )

        # S_0 registered as a buffer so it is saved in checkpoints.
        # PyTorch >= 1.8 supports None buffers (pyproject requires >= 2.3).
        self.register_buffer("S_0", None)  # set on first forward pass
        self.register_buffer("_last_N_active", torch.tensor(0))

        # Sub-modules
        self.spectral_decoder = SpectralLoadingDecoder(
            q=self.dim_latent,
            d=self.dim_input,
        )
        self.vibrational = VibrationalStateBlock(
            dim=self.dim_input,
            n_heads=1,  # feature channel is 1 inside the wrapper
            n_steps=cfg["model"]["n_vibrational_steps"],
            gamma=cfg["model"]["gamma"],
        )
        self.diffusion_decoder = EntropicDiffusionDecoder(
            dim=self.dim_input,
        )

        # Minimal encoder: x -> z for the spectral decoder.
        self._encoder = nn.Linear(self.dim_input, self.dim_latent)

        # Frozen spectral basis and base Laplacian for the smoke path.
        self.register_buffer("U_q", torch.randn(self.dim_input, self.dim_latent))
        self.register_buffer("L_base", torch.eye(self.dim_input))

    # ------------------------------------------------------------------ #

    def _dummy_spectral_metrics(self, L_z: Tensor) -> dict[str, Tensor]:
        """
        Minimal placeholder spectral metrics for the smoke-test path.

        In a full implementation these are derived from the encoder, the
        spectral decoder outputs (omega, S, log_var_S), and the mode KL
        utilities in ``vdeductive.model``.  Here we return positive scalars
        so the clock and ELBO have valid inputs.
        """
        device, dtype = L_z.device, L_z.dtype
        kl_S = torch.tensor(0.01, device=device, dtype=dtype)
        kl_tau = torch.tensor(0.1, device=device, dtype=dtype)
        dark_entropy = torch.tensor(0.05, device=device, dtype=dtype)
        N_active = torch.tensor(self.n_modes // 2, device=device)
        log_lam_s = torch.tensor(0.0, device=device, dtype=dtype)
        rho_plus = torch.tensor(0.5, device=device, dtype=dtype)
        rho_minus = torch.tensor(0.5, device=device, dtype=dtype)
        return {
            "kl_S": kl_S,
            "kl_tau": kl_tau,
            "dark_entropy": dark_entropy,
            "N_active": N_active,
            "log_lam_s": log_lam_s,
            "rho_plus": rho_plus,
            "rho_minus": rho_minus,
        }

    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor) -> ESDMOutput:
        """
        End-to-end forward pass.

        This is a minimal Phase 6 wiring that exercises all sub-modules.  The
        spectral metrics are placeholders; a production model would compute
        them from the encoder and spectral KL utilities.
        """
        # 1. Spectral loading: produce L(z) and placeholder spectral outputs.
        z = self._encoder(x)
        L_z = self.spectral_decoder(z, self.U_q, self.L_base)[3]
        spectral = self._dummy_spectral_metrics(L_z)

        # 2. First-batch S_0 initialisation
        if self.S_0 is None:
            self.S_0 = (
                spectral["kl_S"] + spectral["kl_tau"] + spectral["dark_entropy"]
            ).detach()

        # 3. Entropic clock: update tau, Lambda, delta_tau
        clock_state = self.clock.step(
            kl_tau_current=spectral["kl_tau"],
            phi_current=spectral["log_lam_s"],
            rho_plus=spectral["rho_plus"],
            rho_minus=spectral["rho_minus"],
            dark_entropy=spectral["dark_entropy"],
            S_0=self.S_0,
        )

        # 4. Wave recurrence: Q_t gated by Lambda(tau)
        Q, rho_plus, rho_minus = self.vibrational(
            x,
            L_f=self.L_base,
            Lambda_tau=clock_state.Lambda_tau,
            delta_tau=clock_state.delta_tau,
        )
        self._last_N_active.fill_(int(spectral["N_active"]))

        # 5. Entropic diffusion decode: alpha_bar from clock
        x_hat = self.diffusion_decoder(Q, L_z, self.clock, noise=torch.zeros_like(Q))

        # 6. ELBO
        loss, metrics = esdm_elbo(
            x, x_hat,
            kl_z=torch.tensor(0.0, device=x.device, dtype=x.dtype),
            kl_S=spectral["kl_S"],
            kl_tau=spectral["kl_tau"],
            dark_entropy=torch.tensor(clock_state.dark_entropy, device=x.device, dtype=x.dtype),
            S_0=self.S_0,
        )

        return ESDMOutput(x_hat=x_hat, loss=loss, clock=clock_state, metrics=metrics)

    # ------------------------------------------------------------------ #

    def health_check(self) -> dict:
        """Diagnostic summary — call every N batches, not every step."""
        c = self.clock
        return {
            "tau_n":      float(c.tau_n),
            "tau_flat":   int(c._flat_steps),
            "conserved":  float(c.conserved),
            "rho_sign":   float(c.rho_sign),
            "N_active":   int(self._last_N_active),
            "Lambda_tau": float(c.Lambda_tau),
            "heat_death": c.is_heat_death(),
        }
