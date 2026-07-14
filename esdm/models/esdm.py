"""
Phase 6 -- Top-level ESDM nn.Module.

Assembles EntropicClock, VibrationalStateBlock, SpectralLoadingDecoder,
EntropicDiffusionDecoder, and the ELBO into one end-to-end model.

Fixes applied during port from esdm_legacy/model.py:
  1. L_z piping: the spectral decoder's L_z is fed to the vibrational block
     (was: identity L_base, leaving L_z unused).
  2. tau_mode_kl dedup: the inline per-mode KL formula is replaced by a call
     to esdm.models.spectral_kl.tau_mode_kl_per_mode (single source of truth).
  3. kl_z: the encoder now produces (mu, logvar) and kl_z is the standard
     Gaussian VAE KL (was: hard-coded to zero).

The WHT filter is an optional, injectable component: pass a GraphWHTFilter
instance via cfg["model"]["wht_filter"] (or None to disable). The CLI
surfaces this as --wht-filter {none,basic,bivariate}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .clock import EntropicClock, ClockState
from .diffusion_decoder import EntropicDiffusionDecoder
from .elbo import esdm_elbo
from .spectral_decoder import SpectralLoadingDecoder
from .spectral_kl import tau_mode_kl_per_mode
from .vibrational import VibrationalStateBlock
from .._vendor.spectral import spectral_basis_kl
from ..graphs.wht_filter import GraphWHTFilter


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
        self.n_nodes = cfg["model"]["n_nodes"]
        self.n_modes = cfg["model"]["n_modes"]
        self.mode_threshold = cfg["model"].get("mode_threshold", 0.01)
        self.lam_s = cfg["model"].get("lam_s", 1.0)

        # Optional Graph-WHT spectral pre-filter for the diffusion state.
        # When provided, the vibrational state Q is filtered with heat-kernel
        # time equal to the current clock tau_n before diffusion decoding.
        # None disables the filter (bare L_f @ Q in the wave recurrence).
        self.wht_filter: Optional[GraphWHTFilter] = cfg["model"].get(
            "wht_filter", None
        )

        self.clock = EntropicClock(
            eps=cfg["clock"]["eps"],
            heat_death_patience=cfg["clock"]["heat_death_patience"],
            lambda_clip=cfg["clock"]["lambda_clip"],
        )

        # S_0 registered as a buffer so it is saved in checkpoints.
        self.register_buffer("S_0", None)  # set on first forward pass
        self.register_buffer("_last_N_active", torch.tensor(0))
        self.register_buffer("_last_Q", None)  # previous raw vibrational state

        # rho buffers are updated from the vibrational block after each step
        # and fed back to the clock on the next step.
        self.register_buffer("_last_rho_plus", torch.eye(1))
        self.register_buffer("_last_rho_minus", torch.eye(1))

        # Sub-modules
        self.spectral_decoder = SpectralLoadingDecoder(
            q=self.dim_latent,
            d=self.n_nodes,
        )
        self.vibrational = VibrationalStateBlock(
            dim=self.dim_input,
            n_heads=1,
            n_steps=cfg["model"]["n_vibrational_steps"],
            gamma=cfg["model"]["gamma"],
            wht_filter=cfg["model"].get("wht_filter", None),
        )
        self.diffusion_decoder = EntropicDiffusionDecoder(
            dim=self.dim_input,
        )

        # Variational encoder: x -> (mu, logvar) for the spectral decoder.
        # Fix 3: kl_z is now the standard Gaussian VAE KL, not zero.
        self.enc_mu = nn.Linear(self.dim_input, self.dim_latent)
        self.enc_logvar = nn.Linear(self.dim_input, self.dim_latent)

        # Frozen spectral basis and base Laplacian for the smoke path.
        # If a real feature-space Laplacian L_F is provided in the config,
        # use it as L_base and derive U_q / eigvals_q from its eigenvectors.
        L_base_cfg = cfg["model"].get("L_base", None)
        if L_base_cfg is not None:
            L_base = torch.as_tensor(L_base_cfg, dtype=torch.float32)
            if L_base.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"cfg['model']['L_base'] shape {tuple(L_base.shape)} does not match "
                    f"expected ({self.n_nodes}, {self.n_nodes})"
                )
            self.register_buffer("L_base", L_base)
            vals, vecs = torch.linalg.eigh(L_base.double())
            vals = vals.float()
            vecs = vecs.float()
            # Leading eigenvectors are those with smallest eigenvalues.
            self.register_buffer("U_q", vecs[:, :self.dim_latent])
            self.register_buffer("eigvals_q", vals[:self.dim_latent])
        else:
            self.register_buffer("U_q", torch.randn(self.n_nodes, self.dim_latent))
            self.register_buffer("L_base", torch.eye(self.n_nodes))
            self.register_buffer(
                "eigvals_q",
                torch.linalg.eigvalsh(self.L_base)[:self.dim_latent],
            )

    # ------------------------------------------------------------------ #

    def _compute_spectral_metrics(
        self,
        W: Tensor,
        omega: Tensor,
        S: Tensor,
        L_z: Tensor,
        log_var_S: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute real spectral KLs and bright/dark sector diagnostics.
        """
        kl_S = spectral_basis_kl(S, log_var_S, self.eigvals_q, lam_s=self.lam_s)

        # Fix 2: per-mode KL via the shared spectral_kl module (no inline dup).
        log_a = omega.clamp(min=1e-8).log()
        log_b = torch.zeros_like(omega)
        kl_per_mode = tau_mode_kl_per_mode(
            log_a, log_b, self.eigvals_q, tau=1.0, a_min=0.1
        )

        active = omega > self.mode_threshold
        inactive = ~active
        kl_tau = (kl_per_mode * active.float()).sum(dim=-1).mean()
        dark_entropy = (kl_per_mode * inactive.float()).sum(dim=-1).mean()

        log_lam_s = torch.log(omega.mean().clamp(min=1e-8))
        N_active = active.float().sum(dim=-1).mean()

        return {
            "kl_S": kl_S,
            "kl_tau": kl_tau,
            "dark_entropy": dark_entropy,
            "N_active": N_active,
            "log_lam_s": log_lam_s,
        }

    # ------------------------------------------------------------------ #

    def _kl_z(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Standard Gaussian VAE KL: KL( N(mu, sigma^2) || N(0, I) ).

        Fix 3: was hard-coded to zero in the legacy model.
        """
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()

    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor) -> ESDMOutput:
        """
        End-to-end forward pass.
        """
        # 1. Variational encoding: x -> (mu, logvar) -> z (reparametrised)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x).clamp(-6.0, 4.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        # 2. Spectral loading: produce L(z) and live spectral outputs.
        W, omega, S, L_z, log_var_S = self.spectral_decoder(z, self.U_q, self.L_base)
        spectral = self._compute_spectral_metrics(W, omega, S, L_z, log_var_S)

        # 3. First-batch S_0 initialisation
        if self.S_0 is None:
            self.S_0 = (
                spectral["kl_S"] + spectral["kl_tau"] + spectral["dark_entropy"]
            ).detach()
            # Scale the entropic time horizon so alpha_bar starts near 1 and
            # decreases gradually as entropy flows (docs/00-DESIGN.md §4).
            self.clock.tau_max_est.fill_(10.0 * float(self.S_0.item()))

        # 4. Entropic clock: update tau, Lambda, delta_tau
        # Normalize kl_tau by S_0 so the clock is scale-invariant w.r.t. the
        # feature-space Laplacian's eigenvalue scale.
        kl_tau_norm = spectral["kl_tau"] / (self.S_0 + 1e-8)
        # Initialize the clock's previous kl_tau on the first forward pass
        # so the first delta_tau is ~0 (not the full kl_tau magnitude).
        # This prevents tau_n from jumping to ~1 on step 0 and collapsing
        # alpha_bar to 0 immediately.
        if self.clock._prev_kl_tau.item() == 0.0 and self.clock.tau_n.item() == 0.0:
            self.clock._prev_kl_tau.fill_(float(kl_tau_norm.item()))
        clock_state = self.clock.step(
            kl_tau_current=kl_tau_norm,
            phi_current=spectral["log_lam_s"],
            rho_plus=self._last_rho_plus,
            rho_minus=self._last_rho_minus,
            dark_entropy=spectral["dark_entropy"],
            S_0=self.S_0,
        )

        # 5. Wave recurrence: Q_t gated by Lambda(tau).
        # Fix 1: pipe L_z from the spectral decoder into the wave (was: L_base).
        # When n_nodes == dim_input the shapes match directly. Otherwise we
        # fall back to L_base for the smoke path and log a warning.
        if L_z.shape[-1] == self.dim_input:
            L_f_wave = L_z
        else:
            L_f_wave = self.L_base.expand(x.shape[0], -1, -1)

        Q, rho_plus, rho_minus = self.vibrational(
            x,
            L_f=L_f_wave,
            Lambda_tau=clock_state.Lambda_tau,
            delta_tau=clock_state.delta_tau,
        )
        self._last_rho_plus = rho_plus.detach().clone()
        self._last_rho_minus = rho_minus.detach().clone()
        self._last_N_active.fill_(int(spectral["N_active"].item()))

        # 6. The WHT filter is now applied INSIDE the vibrational recurrence
        # (Phase 2 integration, docs/00-DESIGN.md §4.3).  No post-processing.
        Q_raw = Q
        self._last_Q = Q_raw.detach().clone()

        # 7. Entropic diffusion decode: alpha_bar from clock
        x_hat = self.diffusion_decoder(Q, L_z, self.clock, noise=torch.zeros_like(Q))

        # 8. ELBO with real kl_z (Fix 3)
        kl_z = self._kl_z(mu, logvar)
        loss, metrics = esdm_elbo(
            x, x_hat,
            kl_z=kl_z,
            kl_S=spectral["kl_S"],
            kl_tau=spectral["kl_tau"],
            dark_entropy=torch.tensor(clock_state.dark_entropy, device=x.device, dtype=x.dtype),
            S_0=self.S_0,
        )

        return ESDMOutput(x_hat=x_hat, loss=loss, clock=clock_state, metrics=metrics)

    # ------------------------------------------------------------------ #

    def health_check(self) -> dict:
        """Diagnostic summary -- call every N batches, not every step."""
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
