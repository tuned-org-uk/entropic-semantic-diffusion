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
from esdm.vibrational import VibrationalStateBlock, compute_mode_residuals
from esdm._vendor.spectral import spectral_basis_kl
from esdm.wht_filter import GraphWHTFilter


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
        self.wht_filter: Optional[GraphWHTFilter] = cfg["model"].get(
            "wht_filter", None
        )

        self.clock = EntropicClock(
            eps=cfg["clock"]["eps"],
            heat_death_patience=cfg["clock"]["heat_death_patience"],
            lambda_clip=cfg["clock"]["lambda_clip"],
        )

        # S_0 registered as a buffer so it is saved in checkpoints.
        # PyTorch >= 1.8 supports None buffers (pyproject requires >= 2.3).
        self.register_buffer("S_0", None)  # set on first forward pass
        self.register_buffer("_last_N_active", torch.tensor(0))
        self.register_buffer("_last_Q", None)  # previous raw vibrational state

        # rho buffers are updated from the vibrational block after each step
        # and fed back to the clock on the next step.  Initialised as identity
        # so the first clock.step has well-defined rho_sign diagnostics.
        self.register_buffer("_last_rho_plus", torch.eye(1))
        self.register_buffer("_last_rho_minus", torch.eye(1))

        # Sub-modules
        self.spectral_decoder = SpectralLoadingDecoder(
            q=self.dim_latent,
            d=self.n_nodes,
        )
        self.vibrational = VibrationalStateBlock(
            dim=self.dim_input,
            n_heads=1,  # feature channel is 1 inside the wrapper
            n_steps=cfg["model"]["n_vibrational_steps"],
            gamma=cfg["model"]["gamma"],
        )
        self.use_per_mode = cfg["model"].get("use_per_mode", False)
        self.diffusion_decoder = EntropicDiffusionDecoder(
            dim=self.dim_input,
            use_per_mode=self.use_per_mode,
        )

        # Minimal encoder: x -> z for the spectral decoder.
        self._encoder = nn.Linear(self.dim_input, self.dim_latent)

        # Frozen spectral basis and base Laplacian for the smoke path.
        self.register_buffer("U_q", torch.randn(self.n_nodes, self.dim_latent))
        self.register_buffer("L_base", torch.eye(self.n_nodes))

        # Leading eigenvalues of the base Laplacian, used by spectral KLs.
        # Computed once at init; shape (dim_latent,).
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

        Parameters
        ----------
        W, omega, S, L_z, log_var_S
            Outputs of ``SpectralLoadingDecoder``:
            W (B, N, q), omega (B, q), S (B, q, q), L_z (B, N, N),
            log_var_S (B, q, q).

        Returns
        -------
        dict with keys: kl_S, kl_tau, dark_entropy, N_active, log_lam_s.
        """
        # Spectral-basis KL for the S posterior.
        kl_S = spectral_basis_kl(S, log_var_S, self.eigvals_q, lam_s=self.lam_s)

        # Mode-weight KL: treat omega as the expected value under a Gamma
        # posterior with shape a = omega and rate b = 1.  This gives a live
        # KL that varies with the spectral decoder output while we do not yet
        # have a dedicated variational head for log_a/log_b.
        log_a = omega.clamp(min=1e-8).log()
        log_b = torch.zeros_like(omega)
        kl_per_mode = self._tau_mode_kl_per_mode(log_a, log_b)

        active = omega > self.mode_threshold
        inactive = ~active
        kl_tau = (kl_per_mode * active.float()).sum(dim=-1).mean()
        dark_entropy = (kl_per_mode * inactive.float()).sum(dim=-1).mean()

        # Per-mode tau for the entropic diffusion schedule.  We assign bright
        # modes their own KL and pool dark-mode KL evenly across inactive modes
        # so that every feature dimension has a tau value.
        kl_per_feature = torch.zeros(
            kl_per_mode.shape[0], self.n_nodes,
            device=kl_per_mode.device, dtype=kl_per_mode.dtype
        )
        kl_per_feature[:, :self.dim_latent] = kl_per_mode
        # Dark modes (if any) are pooled as a constant per remaining dimension.
        n_dark_nodes = self.n_nodes - self.dim_latent
        if n_dark_nodes > 0:
            dark_pool = dark_entropy.detach() / max(n_dark_nodes, 1)
            kl_per_feature[:, self.dim_latent:] = dark_pool

        # Clock field: log spectral scale derived from mean mode weight.
        log_lam_s = torch.log(omega.mean().clamp(min=1e-8))

        # Number of active modes (mean over batch).
        N_active = active.float().sum(dim=-1).mean()

        return {
            "kl_S": kl_S,
            "kl_tau": kl_tau,
            "kl_per_mode": kl_per_feature.mean(dim=0),  # (n_nodes,) per-feature tau driver
            "dark_entropy": dark_entropy,
            "N_active": N_active,
            "log_lam_s": log_lam_s,
        }

    def _tau_mode_kl_per_mode(
        self,
        log_a: Tensor,
        log_b: Tensor,
        tau: float = 1.0,
        a_min: float = 0.1,
    ) -> Tensor:
        """
        Per-mode KL( Gamma(a,b) || Exp(tau * lambda_k) ), shape (B, q).

        Mirrors ``esdm._vendor.spectral.tau_mode_kl`` but returns the
        per-mode values so they can be split into bright/active and
        dark/inactive sectors.
        """
        a = log_a.exp().clamp(min=a_min)
        b = log_b.exp()
        r = tau * self.eigvals_q.clamp(min=1e-6).unsqueeze(0)
        kl = (
            log_b
            - r.log()
            - torch.lgamma(a)
            - (1.0 - a) * torch.digamma(a)
            - a
            + a * r / b
        )
        return kl

    # ------------------------------------------------------------------ #

    def forward(self, x: Tensor) -> ESDMOutput:
        """
        End-to-end forward pass.

        Spectral KLs are computed live from the spectral decoder outputs and
        drive the entropic clock, the conservation term, and the diffusion
        schedule.
        """
        # 1. Spectral loading: produce L(z) and live spectral outputs.
        z = self._encoder(x)
        W, omega, S, L_z, log_var_S = self.spectral_decoder(z, self.U_q, self.L_base)
        spectral = self._compute_spectral_metrics(W, omega, S, L_z, log_var_S)

        # 2. First-batch S_0 initialisation
        if self.S_0 is None:
            self.S_0 = (
                spectral["kl_S"] + spectral["kl_tau"] + spectral["dark_entropy"]
            ).detach()

        # 3. Entropic clock: update tau, Lambda, delta_tau
        # When use_per_mode is enabled, drive the clock with per-feature KLs so
        # the diffusion schedule can vary by mode.  Scalar tau_n is still the
        # sum of absolute changes, preserving the monotonic entropic-time
        # invariant.
        kl_tau_driver = (
            spectral["kl_per_mode"]
            if self.use_per_mode
            else spectral["kl_tau"]
        )
        clock_state = self.clock.step(
            kl_tau_current=kl_tau_driver,
            phi_current=spectral["log_lam_s"],
            rho_plus=self._last_rho_plus,
            rho_minus=self._last_rho_minus,
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
        self._last_rho_plus = rho_plus.detach().clone()
        self._last_rho_minus = rho_minus.detach().clone()
        self._last_N_active.fill_(int(spectral["N_active"].item()))

        # 5. Optional timestep-aware Graph-WHT smoothing of the diffusion state.
        Q_raw = Q
        if self.wht_filter is not None:
            Q_prev = (
                self._last_Q
                if self._last_Q is not None
                else torch.zeros_like(Q_raw)
            )
            r_k = compute_mode_residuals(
                Q_raw, Q_prev, bright_modes=self.n_modes
            )
            Q_np = Q_raw.detach().cpu().numpy()
            r_k_np = r_k.detach().cpu().numpy()
            Q_np = self.wht_filter.forward(
                Q_np,
                tau=float(self.clock.tau_n),
                tau_max=float(self.clock.tau_max_est),
                r_k=r_k_np,
                t=float(self.clock.tau_n),
            )
            Q = torch.from_numpy(Q_np).to(device=Q.device, dtype=Q.dtype)

        # Persist the raw vibrational state for the next step's residual.
        self._last_Q = Q_raw.detach().clone()

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
