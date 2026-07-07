"""
Experiment 1 — Entropic Clock Calibration on a Two-Mode Synthetic Graph.

Ref: docs/02-EXPERIMENTS.md §Experiment 1

Note
----
This is a *pure clock unit test*.  The two-community Laplacian is built and
its spectral gap is logged, but the ``EntropicClock`` is driven by injected
scalar ``kl_tau`` / ``phi`` signals.  Graph-to-clock coupling is validated in
later experiments (Exp 2+).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..models.clock import EntropicClock

logger = logging.getLogger(__name__)


def build_two_community_laplacian(
    n_per_community: int = 5,
    bridge_weight: float = 0.0,
) -> np.ndarray:
    """
    Build a 2-community graph Laplacian.

    Each community is a clique of ``n_per_community`` nodes with edge weight
    1.0.  The two communities are joined by a single bridge edge between the
    first node of each community, with weight ``bridge_weight``.

    Parameters
    ----------
    n_per_community : int
        Number of nodes per community.
    bridge_weight : float
        Weight of the single inter-community bridge edge.

    Returns
    -------
    L : np.ndarray, shape (N, N) with N = 2 * n_per_community
        Symmetric graph Laplacian.
    """
    N = 2 * n_per_community
    A = np.zeros((N, N), dtype=np.float64)

    # Within-community cliques.
    for comm in range(2):
        start = comm * n_per_community
        end = start + n_per_community
        for i in range(start, end):
            for j in range(i + 1, end):
                A[i, j] = A[j, i] = 1.0

    # Single bridge edge.
    A[0, n_per_community] = A[n_per_community, 0] = bridge_weight

    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    return L


def kl_tau_signal(step: int) -> float:
    """
    Scripted four-phase KL injection for Experiment 1.

    Parameters
    ----------
    step : int
        1-indexed step in [1, 250].

    Returns
    -------
    kl_tau : float
    """
    if 1 <= step <= 50:
        # Phase A: flat.
        return 1.0
    if 51 <= step <= 150:
        # Phase B: linear rise 1.0 -> 3.0.
        return 1.0 + 2.0 * (step - 50) / 100.0
    if 151 <= step <= 200:
        # Phase C: flat.
        return 3.0
    # Phase D: oscillating (steps 201-250).
    return 3.0 + 0.5 * np.sin((step - 200) * np.pi / 10.0)


def phi_signal(kl_tau: float) -> float:
    """Clock field simulating ``log_lam_s`` from the design doc."""
    return float(np.log(1.0 + kl_tau))


def run_experiment(
    bridge_weight: float = 0.1,
    n_steps: int = 250,
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, EntropicClock]:
    """
    Run Experiment 1 for a single bridge weight.

    Parameters
    ----------
    bridge_weight : float
        Bridge edge weight (w ∈ {0.0, 0.01, 0.1, 1.0} in the sweep).
    n_steps : int
        Number of synthetic training steps.
    seed : int
        Random seed for reproducibility.  The clock itself is deterministic
        given the KL signal, but the seed is logged for traceability.
    output_dir : Path or str, optional
        Directory in which to write ``metrics.csv``.  If None, no CSV is
        written.

    Returns
    -------
    metrics : pd.DataFrame
        Per-step metrics with columns matching the shared logging convention.
    clock : EntropicClock
        The clock instance after the 250 steps, used for success-criterion
        assertions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build Laplacian (used only to set the spectral context; the clock itself
    # is driven by the injected KL signal).
    L = build_two_community_laplacian(
        n_per_community=5,
        bridge_weight=bridge_weight,
    )
    logger.info(
        "Experiment 1: bridge_weight=%.4f, spectral_gap=%.6f",
        bridge_weight,
        np.sort(np.linalg.eigvalsh(L))[1],
    )

    clock = EntropicClock(
        eps=1e-8,
        heat_death_patience=200,
        lambda_clip=100.0,
    )

    # Dummy rho tensors — clock is run in isolation, so constructive/destructive
    # densities are identity (rho_sign = 1.0).
    rho_plus = torch.eye(1, dtype=torch.float64)
    rho_minus = torch.eye(1, dtype=torch.float64)
    dark_entropy = torch.tensor(0.0, dtype=torch.float64)

    records = []
    for step in range(1, n_steps + 1):
        kl_tau = kl_tau_signal(step)
        phi = phi_signal(kl_tau)

        kl_tau_t = torch.tensor(kl_tau, dtype=torch.float64)
        phi_t = torch.tensor(phi, dtype=torch.float64)

        state = clock.step(
            kl_tau_current=kl_tau_t,
            phi_current=phi_t,
            rho_plus=rho_plus,
            rho_minus=rho_minus,
            dark_entropy=dark_entropy,
            S_0=None,
        )

        records.append({
            "step": step,
            "bridge_weight": bridge_weight,
            "seed": seed,
            "spectral_gap": float(np.sort(np.linalg.eigvalsh(L))[1]),
            "kl_tau": kl_tau,
            "phi": phi,
            "tau_n": state.tau_n,
            "delta_tau": state.delta_tau,
            "Lambda_tau": state.Lambda_tau,
            "alpha_bar": float(clock.alpha_bar()),
            # Clock-isolation mode: no active-mode count.  NaN makes the
            # absence explicit for downstream analysis.
            "N_active": float("nan"),
            "conserved": state.conserved,
            "rho_sign": state.rho_sign,
            "elbo": 0.0,
            "kl_z": 0.0,
            "kl_S": 0.0,
            "kl_tau_elbo": 0.0,
            "recon_mse": 0.0,
            "is_heat_death": int(clock.is_heat_death()),
        })

    metrics = pd.DataFrame(records)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics, clock


def assert_success_criteria(
    metrics: pd.DataFrame,
    clock: EntropicClock,
    tol: float = 1e-6,
) -> None:
    """
    Assert Experiment 1 success criteria.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``run_experiment``.
    clock : EntropicClock
        Clock instance used to generate ``metrics``.
    tol : float
        Tolerance for "flat" phases.

    Raises
    ------
    AssertionError
        If any success criterion is violated.
    """
    phase_a = metrics[metrics["step"].between(1, 50)]
    phase_b = metrics[metrics["step"].between(51, 150)]
    phase_c = metrics[metrics["step"].between(151, 200)]
    phase_d = metrics[metrics["step"].between(201, 250)]

    # tau_n flat in phases A and C.
    assert (
        phase_a["tau_n"].max() - phase_a["tau_n"].min()
    ) <= tol, "tau_n drifted during flat Phase A"
    assert (
        phase_c["tau_n"].max() - phase_c["tau_n"].min()
    ) <= tol, "tau_n drifted during flat Phase C"

    # tau_n increases monotonically during Phase B (strictly rising KL).
    assert (
        phase_b["tau_n"].diff().dropna() >= -tol
    ).all(), "tau_n decreased during rising Phase B"

    # tau_n is monotonically non-decreasing by construction (it accumulates
    # |Δkl_tau|).  For Phase D the stronger, semantically meaningful check is
    # net entropic advance over the oscillatory phase.
    assert (
        phase_d["tau_n"].diff().dropna() >= -tol
    ).all(), "tau_n decreased during oscillating Phase D"
    assert (
        phase_d["tau_n"].iloc[-1] > phase_d["tau_n"].iloc[0]
    ), "tau_n did not advance during oscillating Phase D"

    # alpha_bar obeys cosine schedule at tau/tau_max_est = 0.5.
    midpoint = 0.5 * float(clock.tau_max_est)
    alpha_at_mid = float(
        clock.alpha_bar(torch.tensor(midpoint, dtype=torch.float64))
    )
    assert abs(alpha_at_mid - 0.5) <= 1e-6, (
        f"alpha_bar at tau/tau_max=0.5 is {alpha_at_mid}, expected 0.5"
    )

    # Lambda_tau finite and clipped.
    assert metrics["Lambda_tau"].lt(100.0 + tol).all(), (
        "Lambda_tau exceeded lambda_clip"
    )
    assert metrics["Lambda_tau"].ge(0.0).all(), "Lambda_tau became negative"

    # No NaN/Inf in any numeric column.  N_active is intentionally NaN in
    # clock-isolation mode and is excluded from this check.
    numeric = metrics.select_dtypes(include=[np.number]).drop(
        columns=["N_active"], errors="ignore"
    )
    assert np.isfinite(numeric.to_numpy()).all(), "NaN or Inf detected in buffers"

    # Heat death should not fire within 250 steps (patience=200).
    assert metrics["is_heat_death"].sum() == 0, "Heat death fired unexpectedly"


def run_bridge_sweep(
    bridge_weights: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0),
    base_output_dir: Optional[Path | str] = None,
    seed: int = 3407,
) -> dict[float, pd.DataFrame]:
    """
    Run Experiment 1 for all bridge weights in the sweep.

    Parameters
    ----------
    bridge_weights : tuple of float
        Bridge edge weights to sweep.
    base_output_dir : Path or str, optional
        Base directory; each weight gets a subdirectory
        ``runs/exp1/<seed>/w_<weight>/``.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Mapping from bridge weight to metrics DataFrame.
    """
    results = {}
    for w in bridge_weights:
        if base_output_dir is not None:
            out = Path(base_output_dir) / f"w_{w}"
        else:
            out = None
        metrics, clock = run_experiment(
            bridge_weight=w,
            seed=seed,
            output_dir=out,
        )
        assert_success_criteria(metrics, clock)
        results[w] = metrics
        logger.info("Experiment 1 passed for bridge_weight=%.4f", w)
    return results
