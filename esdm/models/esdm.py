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

Issue #20: the vibrational block now receives U_q so the full hybridisation
operator H is applied in the wave recurrence.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .bounded_readout import BoundedSpectralReadout
from .clock import EntropicClock, ClockState
from .diffusion_decoder import EntropicDiffusionDecoder
from .elbo import esdm_elbo
from .spectral_decoder import SpectralLoadingDecoder
from .spectral_kl import tau_mode_kl_per_mode
from .vibrational import VibrationalStateBlock
from .._vendor.spectral import spectral_basis_kl


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
        if self.n_nodes != self.dim_input:
            raise ValueError("n_nodes must equal dim_input for the feature-space wave and readout")
        if not 1 <= self.dim_latent <= self.n_nodes:
            raise ValueError("chart rank must satisfy 1 <= dim_latent <= n_nodes")
        # Mode threshold is derived per-batch as median(omega), not a fixed constant.
        # See 001-IMPLEMENTATION-PLAN.md §3.1.
        self.lam_s = cfg["model"].get("lam_s", 1.0)

        # Derive lambda_clip from F (NOT a hardcoded constant).
        # The pump force at clip: lambda_clip * ||B_t|| ~ lambda_clip (||Q||~1 for L2-normalised data).
        # The H force: eta * ||grad_R|| ~ eta * 2*lambda_max / ||Q||^2 = 0.1*F.
        # For balance at the clip point: lambda_clip = 0.1 * F.
        # This follows the same derivation pattern as eta_init (same 0.1 factor, same F).
        # See 00-DESIGN.md S4.2 for the full derivation.
        lambda_clip_derived = 0.1 * self.dim_input

        self.clock = EntropicClock(
            eps=cfg["clock"]["eps"],
            heat_death_patience=cfg["clock"]["heat_death_patience"],
            lambda_clip=lambda_clip_derived,
            eps_rel=cfg["clock"].get("eps_rel", 1e-3),
        )

        # S_0 registered as a buffer so it is saved in checkpoints.
        self.register_buffer("S_0", None)  # set on first forward pass
        self.register_buffer("_last_N_active", torch.tensor(0))
        self.register_buffer("_last_Q", None)  # previous raw vibrational state
        self.register_buffer("_last_recon", None)  # last reconstruction error

        # rho buffers are updated from the vibrational block after each step
        # and fed back to the clock on the next step.
        self.register_buffer("_last_rho_plus", torch.eye(1))
        self.register_buffer("_last_rho_minus", torch.eye(1))

        # Frozen spectral basis and base Laplacian.
        # Process L_base BEFORE creating the vibrational block so that
        # eta_init can be derived from L_F's spectral scale.
        L_base_cfg = cfg["model"].get("L_base", None)
        if L_base_cfg is not None:
            L_base = torch.as_tensor(L_base_cfg).detach().to(self.clock.tau_n, copy=True)
            if L_base.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"cfg['model']['L_base'] shape {tuple(L_base.shape)} does not match "
                    f"expected ({self.n_nodes}, {self.n_nodes})"
                )
            self.register_buffer("L_base", L_base)
            # Do the one-off factorisation on CPU (MPS has no float64).
            vals, vecs = torch.linalg.eigh(L_base.to(device="cpu", dtype=torch.float64))
            vals = vals.to(L_base)
            vecs = vecs.to(L_base)
            # Leading eigenvectors are those with smallest eigenvalues.
            self.register_buffer("U_q", vecs[:, :self.dim_latent])
            self.register_buffer("eigvals_q", vals[:self.dim_latent])
            # lambda_max = largest eigenvalue of L_F (last in ascending order).
            lambda_max = float(vals[-1].clamp(min=1e-8))
        else:
            self.register_buffer("L_base", torch.eye(self.n_nodes))
            # For the identity graph the coordinate axes form an exact eigenbasis.
            self.register_buffer("U_q", self.L_base[:, :self.dim_latent].clone())
            self.register_buffer("eigvals_q", self.L_base.diagonal()[:self.dim_latent].clone())
            lambda_max = 2.0  # upper bound for normalised Laplacian

        # ArrowSpace lambdas: per-item energy scores from the graph build.
        # Defines the energy distribution on the Laplacian graph.
        lambdas = cfg["model"].get("lambdas", None)
        if lambdas is not None:
            lambdas = torch.as_tensor(lambdas).detach().to(self.L_base, copy=True)
            self.register_buffer("lambdas", lambdas)
        else:
            self.register_buffer("lambdas", None)

        # Derive eta_init from L_F spectral scale (NOT a hardcoded constant).
        # The Rayleigh gradient magnitude scales as 2*lambda_max/||Q||^2.
        # For pi_contraction_ratio ~ 0.1 (healthy range), eta ~ 0.1*F/(2*lambda_max).
        # This makes the H operator's initial strength dataset-agnostic.
        eta_init = 0.1 * self.dim_input / (2.0 * lambda_max)

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
            sigma_sq=cfg["model"].get("sigma_sq", 1.0),
            eta=eta_init,
            freeze_h=cfg["model"].get("freeze_h", False),
        )
        self.diffusion_decoder = EntropicDiffusionDecoder(
            dim=self.dim_input,
        )
        self.readout = BoundedSpectralReadout(self.dim_latent)

        # Variational encoder: x -> (mu, logvar) for the spectral decoder.
        self.enc_mu = nn.Linear(self.dim_input, self.dim_latent)
        self.enc_logvar = nn.Linear(self.dim_input, self.dim_latent)
        # Start in the existing graph chart; keep the map trainable and amplitudes free.
        # See docs/00-DESIGN.md, v2.0.1 encoder initialisation.
        with torch.no_grad():
            self.enc_mu.weight.copy_(self.U_q.T)
            self.enc_mu.bias.zero_()
        self.register_load_state_dict_pre_hook(self._prepare_checkpoint)

    # ------------------------------------------------------------------ #

    def _prepare_checkpoint(self, module, state, prefix, *args) -> None:
        """Check the v2 operator and allocate lazy buffers on the receiving device."""
        logits = state.get(prefix + "readout.logits")
        if logits is None:
            raise ValueError("No v2 readout logits in checkpoint; use the original model or retrain.")
        if not torch.isfinite(logits).all():
            raise ValueError("Checkpoint contains non-finite readout logits.")
        chart = state.get(prefix + "U_q")
        if chart is None or chart.shape != self.U_q.shape or not chart.is_floating_point():
            raise ValueError("Checkpoint must contain the matching frozen readout chart U_q.")
        # Validate off-device once at load; float32 columns stored as double
        # still have float32 roundoff. Allow rounding for lower precision saves.
        tol = max(1e-5, torch.finfo(chart.dtype).eps)
        basis = chart.detach().to(device="cpu", dtype=torch.float64)
        eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
        if not torch.allclose(basis.T @ basis, eye, atol=tol, rtol=tol):
            raise ValueError("Checkpoint readout requires an orthonormal chart.")

        lazy = ("S_0", "_last_Q", "_last_recon", "lambdas", "vibrational.Q_tm1", "clock.rho_sign")
        for key in lazy:
            saved = state.get(prefix + key)
            parent, _, leaf = key.rpartition(".")
            owner = self.get_submodule(parent)
            if saved is not None:
                setattr(owner, leaf, self.L_base.new_empty(saved.shape))
            elif owner is self:
                # Optional buffers are absent from the state_dict before first use.
                setattr(owner, leaf, None)

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

        # Per-batch median threshold: ~half the modes are bright, ~half dark.
        # This ensures dark_entropy > 0 and the two-sector partition is active.
        # See 001-IMPLEMENTATION-PLAN.md §3.1.
        mode_threshold = omega.median(dim=-1, keepdim=True).values
        active = omega > mode_threshold
        inactive = ~active
        kl_tau = (kl_per_mode * active.float()).sum(dim=-1).mean()
        # Dark entropy: the KL sum over inactive modes. Once S_0 is
        # initialised, this is bounded as the residual S_0 - S_bright
        # to structurally enforce conservation (00-DESIGN.md §2.3).
        dark_entropy_raw = (kl_per_mode * inactive.float()).sum(dim=-1).mean()
        if self.S_0 is not None:
            # Bound: dark = max(0, S_0 - kl_S - kl_tau)
            dark_entropy = (self.S_0 - kl_S.detach() - kl_tau.detach()).clamp(min=0.0)
        else:
            dark_entropy = dark_entropy_raw

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

    def encode(
        self, x: Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return posterior mean, log variance and an unprojected Gaussian draw."""
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x).clamp(-6.0, 4.0)
        eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        z = mu + torch.exp(0.5 * logvar) * eps
        return mu, logvar, z

    def forward(self, x: Tensor, *, generator: torch.Generator | None = None) -> ESDMOutput:
        """Encode, evolve the raw wave, and reconstruct its bounded-gain signal."""
        mu, logvar, z = self.encode(x, generator=generator)

        # 2. Spectral loading: produce L(z) and live spectral outputs.
        W, omega, S, L_z, log_var_S = self.spectral_decoder(z, self.U_q, self.L_base)
        spectral = self._compute_spectral_metrics(W, omega, S, L_z, log_var_S)

        # 3. First-batch S_0 initialisation
        if self.S_0 is None:
            self.S_0 = (
                spectral["kl_S"] + spectral["kl_tau"] + spectral["dark_entropy"]
            ).detach()
            # tau_max_est is a fixed dimensionless horizon.  The clock
            # accumulates raw |Δkl_tau| (not normalised by S_0), so the
            # horizon is in nat units.  10 nats of cumulative KL change is
            # sufficient for alpha_bar to complete its cosine decay.
            # tau_max_est = 3 * S_0: derived constant from the entropy budget.
            # With S_0 ~ 52 on CIFAR-10, this gives tau_max ~ 156, spreading
            # the cosine schedule across the full run. See 001-IMPLEMENTATION-PLAN.md §4.1.
            self.clock.tau_max_est.fill_(float(self.S_0 * 3.0))

        # 4. Entropic clock: update tau, Lambda, delta_tau
        # Raw kl_tau (no S_0 normalisation): normalising by S_0 compressed
        # the per-step deltas to ~3e-4 on CIFAR-10 (S_0~52), freezing the
        # clock.  Raw kl_tau gives ~0.01-0.05/step, reaching tau_max_est=10
        # within a few hundred steps.
        kl_tau_raw = spectral["kl_tau"]
        # Initialize the clock's previous kl_tau on the first forward pass
        # so the first delta_tau is ~0 (not the full kl_tau magnitude).
        if self.clock._prev_kl_tau.item() == 0.0 and self.clock.tau_n.item() == 0.0:
            self.clock._prev_kl_tau.fill_(float(kl_tau_raw.item()))
        clock_state = self.clock.step(
            kl_tau_current=kl_tau_raw,
            phi_current=spectral["log_lam_s"],
            rho_plus=self._last_rho_plus,
            rho_minus=self._last_rho_minus,
            dark_entropy=spectral["dark_entropy"],
            S_0=self.S_0,
        )

        # 5. Wave recurrence: Q_t gated by Lambda(tau).
        Q, rho_plus, rho_minus = self.vibrational(
            x,
            L_f=L_z,
            Lambda_tau=clock_state.Lambda_tau,
            delta_tau=clock_state.delta_tau,
            U_q=self.U_q,
        )
        self._last_rho_plus = rho_plus.detach().clone()
        self._last_rho_minus = rho_minus.detach().clone()
        self._last_N_active.fill_(int(spectral["N_active"].item()))

        self._last_Q = Q.detach().clone()

        # 7. Entropic diffusion decode: alpha_bar from clock
        signal = self.readout(Q, self.U_q)
        x_hat = self.diffusion_decoder(signal, L_z, self.clock, noise=torch.zeros_like(Q))

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
        self._last_recon = metrics.get("recon", torch.tensor(1.0)).detach().clone()

        return ESDMOutput(x_hat=x_hat, loss=loss, clock=clock_state, metrics=metrics)

    # ------------------------------------------------------------------ #

    def health_check(self) -> dict:
        """Diagnostic summary -- call every N batches, not every step."""
        c = self.clock
        # Decoder-collapse: recon below 1e-3 means the diffusion decoder
        # has collapsed to a constant output (trivial ELBO optimum).
        last_recon = float(self._last_recon) if self._last_recon is not None else 1.0
        # H-operator diagnostics from the vibrational block.
        h_diag = self.vibrational.diagnostics()
        return {
            "tau_n":            float(c.tau_n),
            "tau_flat":         int(c._flat_steps),
            "conserved":        float(c.conserved),
            "rho_sign":         float(c.rho_sign),
            "N_active":         int(self._last_N_active),
            "Lambda_tau":       float(c.Lambda_tau),
            "heat_death":       c.is_heat_death(),
            "decoder_collapsed": last_recon < 1e-3,
            "angular_displacement": h_diag.get("angular_displacement", 0.0),
            "pi_contraction_ratio": h_diag["pi_contraction_ratio"],
            "gamma_mean": h_diag.get("gamma_mean", 0.0),
        }
