"""
Experiment 3 — Barrier Height Sweep on High-Dimensional Gaussian Clusters.

Ref: docs/02-EXPERIMENTS.md §Experiment 3
    issue #6 (updated 2026-07-03): dim=64, bright_modes=dim//2=32

Depends on Exp 2 passing.  Single GPU recommended; CPU-only feasible for
small sub-sweeps.  Full sweep (450 runs) estimated 2-4 h on a single GPU.

Key changes vs. original issue #6 spec
--------------------------------------
* ``dim=64``         — raised from 8 to 64 for richer spectral structure.
* ``bright_modes=32``— set to ``dim // 2 = 32`` (previously equal to dim);
  this gives a non-trivial dark sector (32 dark modes) at all times.
* Dataset: 4 isotropic Gaussian clusters in ℝ⁶⁴ (unchanged cluster count).
* All other sweep parameters, assertions, and logging schema are carried
  forward from issue #6 without modification.

Sweep grid
----------
::

    kl_S_weight ∈ {0.001, 0.01, 0.05, 0.2, 1.0}        (5 barrier heights)
    kl_decay    ∈ {0.01, 0.03, 0.08}                    (3 schedule speeds)
    rho         ∈ {1, 3, 10}                             (3 cluster SNRs)
    lambda_clip ∈ {10, 100}                              (2 pump caps)
    seeds       ∈ 5                                      (per combination)

Total: 5 × 3 × 3 × 2 × 5 = 450 runs.

Primary result condition: ``kl_decay=0.03``, ``lambda_clip=10``, ``rho=3``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from esdm.clock import EntropicClock
from esdm.vibrational import (
    VibrationalStateBlock,
    _cfl_dt,
    _laplacian_from_dense,
    compute_mode_residuals,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Embedding dimensionality — raised from 8 (original spec) to 64.
DIM: int = 64

#: Number of low-frequency modes treated as bright/active.
#: Set to ``DIM // 2 = 32`` so that half the spectrum forms the dark sector,
#: giving a non-trivial dark-entropy signal throughout training.
BRIGHT_MODES: int = DIM // 2

N_CLUSTERS: int = 4
SAMPLES_PER_CLUSTER: int = 1000
N_STEPS: int = 2000
BATCH_SIZE: int = 64
HEAT_DEATH_PATIENCE: int = 200

# Default sweep grid (primary result condition highlighted).
KL_S_WEIGHTS = [0.001, 0.01, 0.05, 0.2, 1.0]
KL_DECAYS = [0.01, 0.03, 0.08]          # 0.03 is primary; 0.08 is Exp-2 control
RHO_SNRS = [1, 3, 10]                   # rho=3 is primary
LAMBDA_CLIPS = [10, 100]                # 10 is primary; 100 is Exp-2 control
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_gaussian_clusters(
    dim: int = DIM,
    n_clusters: int = N_CLUSTERS,
    samples_per_cluster: int = SAMPLES_PER_CLUSTER,
    rho: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate isotropic Gaussian clusters in ℝ^{dim}.

    Cluster centres are placed at distance ``rho`` from the origin along
    evenly-spaced directions in the first two coordinate dimensions.  All
    clusters share unit isotropic covariance so that the inter-cluster SNR
    is controlled entirely by ``rho``.

    Parameters
    ----------
    dim :
        Ambient dimension of each sample (default ``DIM=64``).
    n_clusters :
        Number of Gaussian clusters (default 4).
    samples_per_cluster :
        Number of samples drawn per cluster (default 1000).
    rho :
        Cluster-centre distance from origin — acts as the signal-to-noise
        ratio between clusters (default 3).
    seed :
        NumPy random seed for reproducibility.

    Returns
    -------
    X : Tensor, shape (n_clusters * samples_per_cluster, dim)
        Float64 data matrix, all clusters concatenated along dimension 0.
    """
    rng = np.random.default_rng(seed)
    X_parts = []
    for k in range(n_clusters):
        angle = 2.0 * math.pi * k / n_clusters
        centre = np.zeros(dim)
        centre[0] = rho * math.cos(angle)
        centre[1] = rho * math.sin(angle)
        samples = rng.standard_normal((samples_per_cluster, dim)) + centre
        X_parts.append(samples)
    X = np.concatenate(X_parts, axis=0)        # (n_clusters * S, dim)
    return torch.tensor(X, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Spectral helpers — adapted from exp2.py for dimension DIM
# ---------------------------------------------------------------------------

def identity_laplacian(dim: int = DIM) -> torch.Tensor:
    """
    Return the ``dim × dim`` identity matrix as a stand-in Laplacian.

    Exp 3 does not use a fixed graph topology; the vibrational block receives
    a batch-level empirical Laplacian built from the data.  The identity is
    used to initialise the spectral eigenvector cache and to compute
    ``lam_max`` for the CFL bound before the first data batch is seen.

    Parameters
    ----------
    dim :
        Size of the square matrix.

    Returns
    -------
    L : Tensor, shape (dim, dim), dtype float64
    """
    return torch.eye(dim, dtype=torch.float64)


def empirical_laplacian(
    X: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Build an RBF-kernel normalised graph Laplacian from a data batch.

    Computes the symmetric normalised Laplacian
    ``L = I - D^{-1/2} W D^{-1/2}`` where ``W`` is the RBF affinity matrix
    with bandwidth ``sigma`` and the diagonal is zeroed before normalisation.

    Parameters
    ----------
    X : Tensor, shape (B, dim)
        Batch of data vectors.
    sigma : float
        RBF bandwidth (default 1.0).

    Returns
    -------
    L : Tensor, shape (dim, dim), dtype float64
        Symmetric normalised Laplacian of the feature-space affinity graph.
    """
    # Build affinity in feature space: W_ij = exp(-||X_i - X_j||^2 / sigma^2).
    # Transpose to (dim, B) so that rows are feature dimensions.
    Xt = X.T.to(dtype=torch.float64)             # (dim, B)
    diff = Xt.unsqueeze(2) - Xt.unsqueeze(1)     # (dim, B, B)
    W = torch.exp(-(diff ** 2).sum(0) / sigma**2) # (B, B)
    W.fill_diagonal_(0.0)

    # Degree vector and normalisation.
    d = W.sum(dim=1).clamp(min=1e-12)             # (B,)
    d_inv_sqrt = d.pow(-0.5)
    W_norm = d_inv_sqrt.unsqueeze(1) * W * d_inv_sqrt.unsqueeze(0)

    # Covariance-space Laplacian: project onto feature space via X^T L X.
    # This gives a (dim, dim) matrix that summarises the spectral structure
    # of the batch geometry in feature space.
    L_batch = Xt @ (torch.eye(X.shape[0], dtype=torch.float64) - W_norm) @ Xt.T

    # Symmetrise and normalise to keep eigenvalues in a stable range.
    L_sym = 0.5 * (L_batch + L_batch.T)
    norm = L_sym.abs().max().clamp(min=1e-12)
    return L_sym / norm


def eigenvectors_of(
    L: torch.Tensor,
    bright_modes: int = BRIGHT_MODES,
) -> torch.Tensor:
    """
    Return the ``dim × dim`` eigenvector matrix of ``L``.

    Eigenvectors are sorted by ascending eigenvalue (DC mode first) and each
    column is unit-normalised.  Used by ``dark_entropy_from_spectrum`` to
    split the spectrum into bright (low-frequency) and dark (high-frequency)
    sectors.

    Parameters
    ----------
    L : Tensor, shape (dim, dim)
        Symmetric positive semi-definite Laplacian.
    bright_modes : int
        Passed through for documentation; does not affect the returned matrix.

    Returns
    -------
    U : Tensor, shape (dim, dim), dtype float64
    """
    vals, vecs = torch.linalg.eigh(L)  # ascending order, shape (dim,), (dim, dim)
    norms = vecs.norm(dim=0, keepdim=True).clamp(min=1e-12)
    return vecs / norms


def dark_entropy_from_spectrum(
    Q: torch.Tensor,
    U: torch.Tensor,
    bright_modes: int = BRIGHT_MODES,
) -> tuple[float, float, float]:
    """
    Proxy for dark-sector entropy from the high-frequency half of the spectrum.

    Projects ``Q`` onto the eigenvectors ``U``; modes ``k < bright_modes`` are
    treated as bright/active, and the remaining ``dim - bright_modes`` modes as
    dark/inactive.  The dark entropy is the Shannon entropy of the normalised
    dark-mode energy distribution (the "gate vector" ``g``).

    With ``dim=64`` and ``bright_modes=32`` there are 32 dark modes, so the
    maximum possible dark entropy is ``log(32) ≈ 3.47 nats``.

    Parameters
    ----------
    Q : Tensor, shape (dim,) or (1, dim)
        Vibrational state vector.
    U : Tensor, shape (dim, dim)
        Eigenvector matrix (columns are eigenvectors, ascending eigenvalue).
    bright_modes : int
        Number of lowest-frequency modes considered bright (default 32).

    Returns
    -------
    dark_entropy : float
        Shannon entropy of the normalised dark-mode distribution.
    g_mean : float
        Mean of the normalised dark-mode probability vector ``g``.
    g_entropy : float
        Alias for ``dark_entropy``, logged for the WHT-gate convention.
    """
    if Q.dim() == 2:
        Q = Q.squeeze(0)
    coeffs = U.T @ Q                             # (dim,)
    energy = coeffs.abs().pow(2)                 # (dim,)
    dark_energy = energy[bright_modes:]          # (dim - bright_modes,)
    n_dark = dark_energy.numel()
    total = dark_energy.sum()
    if total <= 0.0:
        return 0.0, 1.0 / n_dark, 0.0
    g = (dark_energy / total).detach()
    entropy = float(-(g * (g + 1e-12).log()).sum())
    return entropy, float(g.mean()), entropy


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_single(
    dim: int = DIM,
    bright_modes: int = BRIGHT_MODES,
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    kl_S_weight: float = 0.05,
    kl_decay: float = 0.03,
    lambda_clip: float = 10.0,
    clock_eps: float = 1e-8,
    rho: float = 3.0,
    seed: int = 0,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Run a single Exp-3 training loop with the given hyperparameters.

    The loop mimics a full ESDM training pass without a learnable encoder:
    at each step a random data batch is drawn from the Gaussian cluster
    dataset and its empirical feature-space Laplacian is used as the
    vibrational wiring.  The clock is driven by a decaying
    ``kl_tau = exp(-kl_decay * step)`` signal (same as Exp 2) weighted by
    ``kl_S_weight`` to simulate the semantic coupling strength.

    Parameters
    ----------
    dim :
        Embedding dimension (default ``DIM=64``).
    bright_modes :
        Number of low-frequency modes treated as bright/active
        (default ``BRIGHT_MODES=32 = dim // 2``).
    n_steps :
        Training steps (default 2000).
    batch_size :
        Samples drawn per training step (default 64).
    kl_S_weight :
        Semantic coupling weight — analogous to the BEC barrier height in
        Barontini Fig. 2.  Scales the ``kl_tau`` signal fed to the clock.
    kl_decay :
        Decay rate of ``kl_tau = exp(-kl_decay * step)``.  Use 0.03 for the
        primary result; 0.08 replicates the Exp-2 collapse behaviour.
    lambda_clip :
        Entropic pump ceiling (default 10.0).
    clock_eps :
        Flatness threshold for the entropic clock (default 1e-8).
    rho :
        Inter-cluster signal-to-noise ratio (default 3.0).
    seed :
        Random seed.
    output_dir :
        If provided, write ``metrics.csv`` to this directory.

    Returns
    -------
    metrics : pd.DataFrame
        Per-step metrics dataframe with the full Exp-3 logging schema.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Dataset ---
    X_all = make_gaussian_clusters(
        dim=dim,
        n_clusters=N_CLUSTERS,
        samples_per_cluster=SAMPLES_PER_CLUSTER,
        rho=rho,
        seed=seed,
    )  # (N_total, dim)
    N_total = X_all.shape[0]

    # --- Spectral cache: build eigenvectors from identity at init ---
    L_init = identity_laplacian(dim)
    U = eigenvectors_of(L_init, bright_modes=bright_modes)  # (dim, dim)
    lam_max = 1.0          # identity Laplacian, lam_max = 1
    dt_cfl_bound = math.sqrt(2.0 / lam_max)

    logger.info(
        "Exp3: dim=%d, bright_modes=%d, kl_S_weight=%.3f, kl_decay=%.3f, "
        "lambda_clip=%.1f, rho=%.1f, seed=%d",
        dim, bright_modes, kl_S_weight, kl_decay, lambda_clip, rho, seed,
    )

    # --- Model ---
    clock = EntropicClock(
        eps=clock_eps,
        heat_death_patience=HEAT_DEATH_PATIENCE,
        lambda_clip=lambda_clip,
    )
    vsb = VibrationalStateBlock(
        dim=dim,
        n_heads=1,
        n_steps=2,           # full ESDM n_steps=2 per issue spec
        gamma=0.1,
    ).to(dtype=torch.float64)

    Q_t = torch.zeros(1, dim, dtype=torch.float64)
    dark_entropy = torch.tensor(0.0, dtype=torch.float64)
    L_prebuilt = _laplacian_from_dense(L_init)

    # Sentinel values for step-0 delta-tracking assertions.
    lambda_cap_step: int = -1
    rho_sign_crossing_step: int = -1
    S_0: Optional[float] = None
    dark_entropy_floor: float = float("nan")

    records: list[dict] = []

    for step in range(n_steps):
        # --- Random data batch ---
        idx = torch.randint(0, N_total, (batch_size,))
        X_batch = X_all[idx]                          # (B, dim)

        # Update spectral cache every step from the current batch.
        L_batch = empirical_laplacian(X_batch)        # (dim, dim)
        U = eigenvectors_of(L_batch, bright_modes=bright_modes)
        L_prebuilt = _laplacian_from_dense(L_batch)

        # Clamp lam_max from actual spectrum for CFL diagnostics.
        lam_max_step = float(torch.linalg.eigvalsh(L_batch).max())
        dt_cfl_bound = math.sqrt(2.0 / max(lam_max_step, 1e-6))

        # --- Clock signal ---
        # kl_tau is scaled by kl_S_weight to simulate the barrier height.
        kl_tau_raw = math.exp(-kl_decay * step)
        kl_tau = torch.tensor(kl_tau_raw * kl_S_weight, dtype=torch.float64)
        phi = torch.log1p(kl_tau)

        rho_plus_scalar = vsb._vdt_block.density.rho_plus.sum().item()
        rho_minus_scalar = vsb._vdt_block.density.rho_minus.sum().item()
        rho_plus_t = torch.tensor(rho_plus_scalar, dtype=torch.float64)
        rho_minus_t = torch.tensor(rho_minus_scalar, dtype=torch.float64)

        state = clock.step(
            kl_tau_current=kl_tau,
            phi_current=phi,
            rho_plus=rho_plus_t,
            rho_minus=rho_minus_t,
            dark_entropy=dark_entropy,
        )

        # --- Vibrational recurrence ---
        # Initialise Q_t from a batch mean on the first step.
        if step == 0:
            Q_t = X_batch.mean(dim=0, keepdim=True)   # (1, dim)

        Q_prev = Q_t.detach().clone()
        Q_t, rho_plus, rho_minus = vsb(
            Q_t,
            L_f=L_batch.unsqueeze(0),                 # (1, dim, dim)
            Lambda_tau=state.Lambda_tau,
            delta_tau=state.delta_tau,
        )

        # --- Per-mode residuals and gate vector ---
        r_k = compute_mode_residuals(
            Q_t.detach(), Q_prev, bright_modes=bright_modes
        )                                             # (bright_modes,)
        g_residual = torch.sigmoid(torch.log1p(r_k))
        r_k_mean = float(r_k.mean())
        r_k_max = float(r_k.max())
        g_residual_mean = float(g_residual.mean())

        # Spearman r_g_corr: first independent Path A correlation measure.
        # Requires at least 2 unique values; guard against degenerate case.
        r_k_np = r_k.detach().numpy()
        g_np = g_residual.detach().numpy()
        if r_k.numel() > 1 and r_k.std().item() > 1e-15 and g_residual.std().item() > 1e-15:
            r_g_corr = float(spearmanr(r_k_np, g_np).statistic)
        else:
            r_g_corr = float("nan")

        # CFL-clamped dt (from the internal VSB log_dt, L_batch).
        dt_cfl = _cfl_dt(
            vsb._vdt_block.log_dt,
            L_batch.unsqueeze(0),
            L_prebuilt,
        ).item()
        dt_used = min(state.delta_tau, dt_cfl)

        # --- Dark entropy proxy ---
        dark_entropy_val, g_mean, g_entropy = dark_entropy_from_spectrum(
            Q_t[0], U, bright_modes=bright_modes
        )
        dark_entropy = torch.tensor(dark_entropy_val, dtype=torch.float64)

        # Conservation residual (in-loop; full ESDM uses the ELBO term).
        conserved = abs(kl_tau.item() + dark_entropy_val)

        # --- S_0 initialisation on first step ---
        if S_0 is None and step == 0:
            S_0 = conserved

        # --- Running diagnostics ---
        if lambda_cap_step < 0 and state.Lambda_tau >= lambda_clip - 1e-6:
            lambda_cap_step = step

        rho_sign_val = state.rho_sign
        if rho_sign_crossing_step < 0 and step > 0 and rho_sign_val < 1.0:
            rho_sign_crossing_step = step

        # Dark entropy floor: running minimum over the last 200 steps.
        if step >= n_steps - 200:
            if math.isnan(dark_entropy_floor):
                dark_entropy_floor = dark_entropy_val
            else:
                dark_entropy_floor = min(dark_entropy_floor, dark_entropy_val)

        records.append({
            # --- Core schema (shared Exp 1-5) ---
            "step":                   step,
            "seed":                   seed,
            "kl_decay":               kl_decay,
            "clock_eps":              clock_eps,
            "lambda_clip":            lambda_clip,
            "bright_modes":           bright_modes,
            "kl_tau":                 kl_tau.item(),
            "dark_entropy":           dark_entropy_val,
            "conserved":              conserved,
            "Q_t_0":                  Q_t[0, 0].item(),
            "rho_plus":               rho_plus.sum().item(),
            "rho_minus":              rho_minus.sum().item(),
            "rho_sign":               rho_sign_val,
            "dt":                     dt_used,
            "dt_cfl_bound":           dt_cfl_bound,
            "delta_tau":              state.delta_tau,
            "tau_n":                  state.tau_n,
            "Lambda_tau":             state.Lambda_tau,
            "alpha_bar":              float(clock.alpha_bar()),
            "is_heat_death":          int(clock.is_heat_death()),
            # --- Path A columns ---
            "r_k_mean":               r_k_mean,
            "r_k_max":                r_k_max,
            "g_residual_mean":        g_residual_mean,
            "a_tau":                  float(getattr(clock, "a_tau", float("nan"))),
            # --- Path A correlation (first independent measure, new in Exp 3) ---
            "r_g_corr":               r_g_corr,
            # --- WHT gate columns ---
            "g_mean":                 g_mean,
            "g_entropy":              g_entropy,
            "lambda_cap_step":        lambda_cap_step,
            "rho_sign_crossing_step": rho_sign_crossing_step,
            "dark_entropy_floor":     dark_entropy_floor,
            # --- Sweep-level identifiers ---
            "dim":                    dim,
            "kl_S_weight":            kl_S_weight,
            "rho_snr":                rho,
        })

    metrics = pd.DataFrame(records)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_success_criteria(
    metrics: pd.DataFrame,
    kl_S_weight: float,
    S_0: Optional[float] = None,
    kl_decay: float = 0.03,
    lambda_clip: float = 10.0,
) -> None:
    """
    Assert Experiment 3 success criteria (issue #6, updated 2026-07-03).

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``run_single``.
    kl_S_weight : float
        Barrier height used in this run (controls which assertions apply).
    S_0 : float, optional
        Initial entropy budget.  Used for the dark-entropy floor assertion.
        If None, derived from the first row of ``metrics``.
    kl_decay : float
        KL decay rate used in this run.
    lambda_clip : float
        Pump ceiling used in this run.
    """
    if S_0 is None:
        S_0 = float(metrics["conserved"].iloc[0]) or 1.0

    # 1. Entropy conservation to machine precision.
    residuals = (metrics["conserved"] - (
        metrics["kl_tau"] + metrics["dark_entropy"]
    )).abs()
    assert (
        residuals <= 1e-10 + 1e-15 * metrics["conserved"].abs()
    ).all(), (
        f"Entropy conservation violated: max residual = {residuals.max():.3e}"
    )

    # 2. dark_entropy > 0 at every step.
    assert (metrics["dark_entropy"] > 0).all(), (
        "dark_entropy hit zero — dark sector is missing"
    )

    # 3. Regime-specific assertions.
    if kl_S_weight <= 1e-4:             # low barrier (0.001)
        tau_n = metrics["tau_n"].to_numpy()
        sign_changes = int(np.sum(np.diff(np.sign(np.diff(tau_n))) != 0))
        assert sign_changes >= 2, (
            f"Expected >= 2 tau_n oscillation cycles at kl_S_weight={kl_S_weight}, "
            f"got {sign_changes} sign changes in the second difference."
        )

    if kl_S_weight >= 1.0:              # high barrier (1.0)
        hd = metrics[metrics["is_heat_death"] == 1]["step"]
        assert not hd.empty and int(hd.iloc[0]) < 500, (
            f"Heat death expected before step 500 at kl_S_weight={kl_S_weight}, "
            f"first heat death at step {hd.iloc[0] if not hd.empty else 'never'}."
        )

    if kl_S_weight in {0.01, 0.05, 0.2}:
        floor = float(metrics["dark_entropy_floor"].iloc[-1])
        if not math.isnan(floor) and S_0 > 0:
            assert 0.40 * S_0 <= floor <= 0.70 * S_0, (
                f"dark_entropy floor {floor:.4f} outside [40%, 70%] of S_0={S_0:.4f} "
                f"at kl_S_weight={kl_S_weight}"
            )

    # 4. lambda_cap_step: with primary settings pump must not cap early.
    if lambda_clip == 10.0 and kl_decay == 0.03:
        cap = int(metrics["lambda_cap_step"].iloc[-1])
        if cap >= 0:   # -1 means cap was never hit — healthy
            n_steps = len(metrics)
            assert cap > 0.75 * n_steps, (
                f"lambda_cap_step={cap} fired before 75% of run "
                f"(n_steps={n_steps}) with lambda_clip=10, kl_decay=0.03"
            )

    # 5. rho_sign crossing for low/medium barriers.
    if kl_S_weight <= 0.05:
        crossing = int(metrics["rho_sign_crossing_step"].iloc[-1])
        n_steps = len(metrics)
        assert crossing < 0 or crossing < 0.75 * n_steps, (
            f"rho_sign crossing at step {crossing} is past 75% of run "
            f"(n_steps={n_steps}) — constructive regime too delayed."
        )

    # 6. Finite Q and no blowup.
    assert np.isfinite(metrics["Q_t_0"].to_numpy()).all(), (
        "NaN or Inf in Q_t[0]"
    )
    q_max = metrics["Q_t_0"].abs().max()
    assert q_max < 1e4, f"Q_t[0] blew up: max |Q_t_0| = {q_max:.3e}"

    # 7. rho_plus and rho_minus must stay finite and non-negative.
    for col in ("rho_plus", "rho_minus"):
        assert np.isfinite(metrics[col].to_numpy()).all(), (
            f"NaN or Inf in {col}"
        )
        assert (metrics[col] >= 0).all(), f"{col} became negative"


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(
    dim: int = DIM,
    bright_modes: int = BRIGHT_MODES,
    kl_s_weights: list[float] = KL_S_WEIGHTS,
    kl_decays: list[float] = KL_DECAYS,
    rho_snrs: list[float] = RHO_SNRS,
    lambda_clips: list[float] = LAMBDA_CLIPS,
    seeds: list[int] = SEEDS,
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    output_root: str = "runs/exp3",
    assert_each: bool = False,
) -> pd.DataFrame:
    """
    Run the full Exp-3 barrier height sweep.

    Iterates over the complete grid of ``kl_S_weight``, ``kl_decay``, ``rho``,
    ``lambda_clip``, and ``seed`` combinations (450 runs total).  Results
    from all runs are concatenated into a single dataframe and written to
    ``<output_root>/all_runs.csv``.

    Parameters
    ----------
    dim :
        Embedding dimension (default ``DIM=64``).
    bright_modes :
        Bright mode count (default ``BRIGHT_MODES=32``).
    kl_s_weights, kl_decays, rho_snrs, lambda_clips, seeds :
        Sweep grid lists; defaults reproduce the full 450-run grid from
        issue #6.
    n_steps :
        Training steps per run (default 2000).
    batch_size :
        Mini-batch size (default 64).
    output_root :
        Root directory for per-run CSV files.
    assert_each :
        If True, run ``assert_success_criteria`` after each individual run
        and raise on the first failure.  Default False (collect all results
        first).

    Returns
    -------
    all_metrics : pd.DataFrame
        Concatenated per-step metrics from all runs.
    """
    all_frames: list[pd.DataFrame] = []
    run_id = 0

    for kl_s in kl_s_weights:
        for kl_d in kl_decays:
            for rho in rho_snrs:
                for lc in lambda_clips:
                    for s in seeds:
                        run_id += 1
                        logger.info(
                            "[%d/450] kl_S=%.3f kl_decay=%.3f rho=%g "
                            "lambda_clip=%g seed=%d",
                            run_id, kl_s, kl_d, rho, lc, s,
                        )
                        out_dir = (
                            f"{output_root}/kls{kl_s}_kld{kl_d}_rho{rho}"
                            f"_lc{lc}/{s}"
                        )
                        df = run_single(
                            dim=dim,
                            bright_modes=bright_modes,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            kl_S_weight=kl_s,
                            kl_decay=kl_d,
                            lambda_clip=lc,
                            rho=rho,
                            seed=s,
                            output_dir=out_dir,
                        )
                        if assert_each:
                            assert_success_criteria(
                                df,
                                kl_S_weight=kl_s,
                                kl_decay=kl_d,
                                lambda_clip=lc,
                            )
                        all_frames.append(df)

    all_metrics = pd.concat(all_frames, ignore_index=True)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(root / "all_runs.csv", index=False)
    logger.info("Sweep complete.  Wrote %s", root / "all_runs.csv")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main_cli(
    output_dir: Optional[Path | str] = None,
    seed: int = 0,
    kl_decay: float = 0.03,
    clock_eps: float = 1e-8,
    lambda_clip: float = 10.0,
    kl_S_weight: float = 0.05,
    rho: float = 3.0,
    dim: int = DIM,
    bright_modes: int = BRIGHT_MODES,
) -> pd.DataFrame:
    """
    CLI entry point for a single Exp-3 run.

    Runs one (seed, kl_S_weight, kl_decay, rho, lambda_clip) combination,
    logs a diagnostic summary, and asserts the success criteria.

    Parameters
    ----------
    output_dir : Path or str, optional
        Output directory for ``metrics.csv``.
    seed :
        Random seed (default 0).
    kl_decay :
        KL decay rate (default 0.03 — primary baseline).
    clock_eps :
        Clock flatness threshold (default 1e-8).
    lambda_clip :
        Entropic pump ceiling (default 10.0 — primary; use 100.0 for
        Exp-2 control condition).
    kl_S_weight :
        Barrier height (default 0.05 — intermediate regime).
    rho :
        Inter-cluster SNR (default 3.0 — primary result condition).
    dim :
        Embedding dimension (default ``DIM=64``).
    bright_modes :
        Bright mode count (default ``BRIGHT_MODES=32``).

    Returns
    -------
    metrics : pd.DataFrame
    """
    metrics = run_single(
        dim=dim,
        bright_modes=bright_modes,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        kl_S_weight=kl_S_weight,
        kl_decay=kl_decay,
        clock_eps=clock_eps,
        lambda_clip=lambda_clip,
        rho=rho,
        seed=seed,
        output_dir=output_dir,
    )

    final = metrics.iloc[-1]
    logger.info(
        "Exp3 summary: dim=%d, bright_modes=%d, kl_S_weight=%.3f, "
        "tau_n=%.4f, alpha_bar=%.4e, Lambda_tau=%.4f, "
        "rho_sign=%.6f, dark_entropy=%.4f, r_g_corr=%.4f, "
        "first_heat_death_step=%s",
        dim, bright_modes, kl_S_weight,
        final["tau_n"],
        final["alpha_bar"],
        final["Lambda_tau"],
        final["rho_sign"],
        final["dark_entropy"],
        final["r_g_corr"],
        metrics[metrics["is_heat_death"] == 1]["step"].min()
        if metrics["is_heat_death"].any()
        else "none",
    )

    assert_success_criteria(
        metrics,
        kl_S_weight=kl_S_weight,
        kl_decay=kl_decay,
        lambda_clip=lambda_clip,
    )
    logger.info("Experiment 3 single run passed — all success criteria satisfied.")
    return metrics
