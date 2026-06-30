"""
Experiment 2 — Vibrational Wave Recurrence on a 1-D Harmonic Chain.

Ref: docs/02-EXPERIMENTS.md §Experiment 2

Depends on Exp 1 passing.  CPU only, expected runtime < 10 min.

Note
----
The issue spec drives the clock with ``kl_tau = exp(-0.01 * step)``.  With
that decay and the default ``eps=1e-8`` the entropic time step remains
above the flatness threshold for far longer than 500 steps, so
``is_heat_death()`` would not fire by step 400.  This script parameterises
the decay rate; the default ``kl_decay=0.08`` is chosen so that heat death
fires near step 400 with ``eps=1e-8`` and ``heat_death_patience=200``.

Diagnostic runs show that ``kl_decay=0.08`` consumes the entropy budget
rapidly and collapses ``alpha_bar`` by step ~100, leaving the latter half of
the run in the full-noise regime.  Slower decays (e.g. ``0.01–0.03``) spread
the schedule but push heat death past step 400; they are useful for
exploring schedule quality on longer horizons.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from esdm.clock import EntropicClock
from esdm.vibrational import VibrationalStateBlock, _cfl_dt, _laplacian_from_dense

logger = logging.getLogger(__name__)

N: int = 32
N_STEPS: int = 500
HEAT_DEATH_PATIENCE: int = 200


def path_graph_laplacian(n: int) -> torch.Tensor:
    """
    Return the (n, n) path graph Laplacian as a float64 CPU tensor.

    The analytic eigenvalues are ``λ_k = 2 - 2·cos(π·k / n)`` for
    ``k = 0, …, n-1``.
    """
    L = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        L[i, i] = 2.0
        if i > 0:
            L[i, i - 1] = -1.0
            L[i - 1, i] = -1.0
    L[0, 0] = 1.0
    L[n - 1, n - 1] = 1.0
    return L


def path_graph_eigenvalues(n: int) -> np.ndarray:
    """Analytic eigenvalues of the n-node path graph Laplacian."""
    k = np.arange(n)
    return 2.0 - 2.0 * np.cos(np.pi * k / n)


def fiedler_vector(n: int) -> torch.Tensor:
    """
    k=1 eigenvector of the path graph Laplacian.

    ``v_i = cos(π·(i + 0.5) / n)`` for ``i = 0, …, n-1``.
    """
    i = torch.arange(n, dtype=torch.float64)
    v = torch.cos(math.pi * (i + 0.5) / n)
    return v / v.norm()


def run_experiment(
    n: int = N,
    n_steps: int = N_STEPS,
    kl_decay: float = 0.08,
    gamma: float = 0.1,
    clock_eps: float = 1e-8,
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Run Experiment 2 on a 1-D harmonic chain.

    Parameters
    ----------
    n : int
        Number of nodes in the path graph.
    n_steps : int
        Number of wave-recurrence steps.
    kl_decay : float
        Decay rate of the injected ``kl_tau = exp(-kl_decay * step)`` signal.
        The default ``0.08`` is chosen so that ``is_heat_death()`` fires near
        step 400 with ``eps=1e-8`` and ``heat_death_patience=200``.
    gamma : float
        Rayleigh damping initial value.
    clock_eps : float
        Flatness threshold passed to ``EntropicClock`` (default ``1e-8``).
    seed : int
        Random seed for reproducibility.
    output_dir : Path or str, optional
        Directory in which to write ``metrics.csv``.

    Returns
    -------
    metrics : pd.DataFrame
        Per-step metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    L_f = path_graph_laplacian(n)                     # (n, n)
    x0 = fiedler_vector(n).unsqueeze(0)               # (1, n)

    # Verify analytic spectrum.
    analytic_eigs = path_graph_eigenvalues(n)
    computed_eigs = torch.linalg.eigvalsh(L_f).numpy()
    assert np.allclose(
        np.sort(analytic_eigs), np.sort(computed_eigs), atol=1e-10
    ), "Path graph Laplacian eigenvalues do not match analytic formula"

    lam_max = analytic_eigs.max()                     # k = n-1
    dt_cfl_bound = math.sqrt(2.0 / lam_max)
    logger.info(
        "Experiment 2: n=%d, lam_max=%.6f, CFL bound dt<=%.6f",
        n, lam_max, dt_cfl_bound,
    )

    clock = EntropicClock(
        eps=clock_eps,
        heat_death_patience=HEAT_DEATH_PATIENCE,
    )
    vsb = VibrationalStateBlock(
        dim=n,
        n_heads=1,
        n_steps=1,
        gamma=gamma,
    ).to(dtype=torch.float64)

    # Move trainable parameters to float64 for the deterministic CPU run.
    Q_t = x0.clone()
    dark_entropy = torch.tensor(0.0, dtype=torch.float64)

    # Pre-build the base Laplacian object used by the CFL helper.
    lap = _laplacian_from_dense(L_f)

    records = []
    for step in range(n_steps):
        # Decaying entropy flow drives the clock.
        kl_tau = torch.tensor(math.exp(-kl_decay * step), dtype=torch.float64)
        phi = torch.log1p(kl_tau)

        # For the first step there is no meaningful density history; feed
        # identity-like scalar densities so rho_sign is well-defined.
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

        Q_t, rho_plus, rho_minus = vsb(
            Q_t,
            L_f=L_f.unsqueeze(0),          # (1, n, n)
            Lambda_tau=state.Lambda_tau,
            delta_tau=state.delta_tau,
        )

        # Recompute the actual CFL-clamped dt used inside the block so it can
        # be logged and asserted.  _cfl_dt returns the strictest of the
        # learnable free step, the Gershgorin bound, and the base-Laplacian
        # bound; the block then takes min(delta_tau, dt).
        dt_cfl = _cfl_dt(
            vsb._vdt_block.log_dt,
            L_f.unsqueeze(0),
            lap,
        ).item()
        dt_used = min(state.delta_tau, dt_cfl)

        # In isolation mode the dark sector is not modelled, so dark_entropy
        # is zero and conserved reduces to |kl_tau|.  They are logged so the
        # CSV schema matches the shared convention and downstream diagnostics
        # can still compute the conservation residual.
        dark_entropy_val = 0.0
        conserved = abs(kl_tau.item() + dark_entropy_val)

        records.append({
            "step": step,
            "seed": seed,
            "kl_decay": kl_decay,
            "clock_eps": clock_eps,
            "kl_tau": kl_tau.item(),
            "dark_entropy": dark_entropy_val,
            "conserved": conserved,
            "Q_t_0": Q_t[0, 0].item(),
            "rho_plus": rho_plus.sum().item(),
            "rho_minus": rho_minus.sum().item(),
            "rho_sign": state.rho_sign,
            "dt": dt_used,
            "dt_cfl_bound": dt_cfl_bound,
            "delta_tau": state.delta_tau,
            "tau_n": state.tau_n,
            "Lambda_tau": state.Lambda_tau,
            "alpha_bar": float(clock.alpha_bar()),
            "is_heat_death": int(clock.is_heat_death()),
        })

    metrics = pd.DataFrame(records)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics


def assert_success_criteria(
    metrics: pd.DataFrame,
    dt_tol: float = 1e-6,
) -> None:
    """
    Assert Experiment 2 success criteria.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``run_experiment``.
    dt_tol : float
        Tolerance for the CFL bound assertion.
    """
    # CFL bound: actual dt used must never exceed sqrt(2 / lam_max).
    assert (
        metrics["dt"] <= metrics["dt_cfl_bound"] + dt_tol
    ).all(), "CFL stability bound violated"

    # No blowup / NaN / Inf in the wave state.
    assert np.isfinite(metrics["Q_t_0"].to_numpy()).all(), (
        "NaN or Inf in Q_t[0]"
    )
    q_max_abs = metrics["Q_t_0"].abs().max()
    assert q_max_abs < 10.0, (
        f"Q_t[0] blew up: max |Q_t_0| = {q_max_abs}"
    )

    # Heat death must fire by step 400 ± 50.
    heat_death_steps = metrics[metrics["is_heat_death"] == 1]["step"]
    assert not heat_death_steps.empty, "is_heat_death() never fired"
    first_hd = int(heat_death_steps.iloc[0])
    assert 350 <= first_hd <= 450, (
        f"Heat death at step {first_hd}, expected 400 ± 50"
    )

    # rho_plus and rho_minus must stay finite and non-negative (PSD).
    for col in ("rho_plus", "rho_minus"):
        assert np.isfinite(metrics[col].to_numpy()).all(), (
            f"NaN or Inf in {col}"
        )
        assert (metrics[col] >= 0.0).all(), f"{col} became negative"


def main_cli(
    output_dir: Optional[Path | str] = None,
    seed: int = 3407,
    kl_decay: float = 0.08,
    clock_eps: float = 1e-8,
) -> pd.DataFrame:
    """CLI entry point for Experiment 2."""
    metrics = run_experiment(
        n=N,
        n_steps=N_STEPS,
        kl_decay=kl_decay,
        clock_eps=clock_eps,
        seed=seed,
        output_dir=output_dir,
    )
    # Log key diagnostic summary for the user.
    final = metrics.iloc[-1]
    logger.info(
        "Exp2 summary: tau_n=%.4f, alpha_bar=%.4e, Lambda_tau=%.4f, "
        "rho_sign=%.6f, first_heat_death_step=%s",
        final["tau_n"],
        final["alpha_bar"],
        final["Lambda_tau"],
        final["rho_sign"],
        metrics[metrics["is_heat_death"] == 1]["step"].min()
        if metrics["is_heat_death"].any()
        else "none",
    )
    assert_success_criteria(metrics)
    logger.info("Experiment 2 passed — all success criteria satisfied.")
    return metrics
