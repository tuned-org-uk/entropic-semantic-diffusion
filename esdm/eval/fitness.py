"""
ESDM fitness evaluation -- composite health score from docs/00-DESIGN.md §8.1.

The thresholds below were calibrated for the pre-v2 unit-sphere recurrence.
On v2 they are historical diagnostics, not a validated quality metric for
unconstrained wave amplitudes or evidence about the learned reconstruction gains.

Runs exp4/exp5 training and measures health criteria:
- entropic-clock motion (tau_monotone, tau_range_bonus)
- alpha_bar schedule decay
- conservation of total entropy
- heat death detection and timing
- rho-sign stabilization
- loss decrease
- net_energy_flow balance (issue #20: Lambda vs H equilibrium)
- pi_contraction_ratio (issue #20: Pi operator activity)

Usage:
    python -m esdm.eval.fitness [--seed N] [--n-steps N] [exp3 params...]
    python -m esdm.eval.fitness --json  # output as JSON
"""
from __future__ import annotations
import argparse
import json
from typing import Optional
import numpy as np
import pandas as pd

def compute_fitness_real(metrics: pd.DataFrame) -> dict:
    """
    Compute a real-data fitness score from Exp-4/5 metrics.

    Scores training health: entropic-clock motion, conservation,
    reconstruction progress, heat-death behaviour, and the hybridisation
    operator diagnostics (net_energy_flow, pi_contraction_ratio).
    """
    scores: dict[str, float] = {}
    n = len(metrics)
    if n == 0:
        return {"total": 0.0, "criteria": {}}

    # Decoder-collapse guard: if reconstruction error is below 1e-3 the
    # diffusion decoder has found the trivial constant-output optimum.
    # Score the run as 0 so the autoresearch loop discards it rather than
    # rewarding the low loss (which is an artefact of recon -> 0).
    if "recon" in metrics.columns:
        final_recon = float(metrics["recon"].iloc[-1])
        if final_recon < 1e-3:
            return {
                "total": 0.0,
                "criteria": {"decoder_collapsed": 0.0},
                "decoder_collapsed": True,
                "final_recon": final_recon,
            }

    tau = metrics["tau_n"].to_numpy()
    tau_range = float(tau[-1] - tau[0])
    scores["tau_monotone"] = 10.0 if np.all(np.diff(tau) >= -1e-12) else 0.0
    scores["tau_range_bonus"] = min(5.0, tau_range * 50.0)

    ab = metrics["alpha_bar"].to_numpy()
    ab_drop = float(ab[0] - ab[-1])
    scores["alpha_bar_decrease"] = min(10.0, ab_drop * 20.0)

    # Conservation: total entropy should stay close to the initial budget S_0.
    if {"kl_S", "kl_tau", "dark_entropy"}.issubset(metrics.columns):
        total = (
            metrics["kl_S"].to_numpy()
            + metrics["kl_tau"].to_numpy()
            + metrics["dark_entropy"].to_numpy()
        )
    else:
        total = metrics["conserved"].to_numpy()
    S_0 = float(total[0])
    residual = np.abs(total - S_0)
    conserved_ratio = float(residual.mean()) / (abs(S_0) + 1e-8)
    scores["conservation"] = max(0.0, 15.0 * (1.0 - min(1.0, conserved_ratio)))

    # Heat death.
    hd_col = "is_heat_death" if "is_heat_death" in metrics.columns else "heat_death"
    hd = int(metrics[hd_col].sum())
    scores["heat_death"] = 10.0 if hd > 0 else 0.0
    if hd > 0:
        hd_step = int(metrics[metrics[hd_col] == 1]["step"].iloc[0])
        # Use planned n_steps (not actual run length) for the fraction,
        # since the run stops when heat death fires.
        n_planned = int(metrics["n_steps"].iloc[0]) if "n_steps" in metrics.columns else n
        frac = hd_step / n_planned
        scores["heat_death_timing"] = max(0.0, 5.0 * (1.0 - abs(frac - 0.7) * 2))

    # Rho-sign stabilisation.
    rs = metrics["rho_sign"].to_numpy()
    if n > 20:
        first_var = float(np.var(rs[: n // 3]))
        last_var = float(np.var(rs[-n // 3 :]))
        scores["rho_sign_stabilizing"] = 10.0 if last_var < first_var else 0.0

    # Reconstruction quality during the active diffusion phase.
    # Measures median recon where alpha_bar > 0.01 (decoder is challenged).
    # In the historical identity readout, alpha_bar=1 gave x_hat=Q. In v2 it
    # gives the bounded readout of Q, which need not equal x. At alpha_bar=0,
    # x_hat=0 and recon = ||x||^2/F. The meaningful range is in between.
    # Lower median = better decoder. Score: 10 pts if median < 0.001,
    # linearly decreasing to 0 at median > 0.01.
    if "recon" in metrics.columns:
        recon = metrics["recon"].to_numpy()
        ab = metrics["alpha_bar"].to_numpy() if "alpha_bar" in metrics.columns else None
        if ab is not None:
            active = recon[ab > 0.01]
        else:
            active = recon
        if len(active) > 0:
            recon_median = float(np.median(active))
            scores["recon_decrease"] = max(0.0, min(10.0, 10.0 * (1.0 - recon_median / 0.01)))
        else:
            scores["recon_decrease"] = 0.0
    else:
        scores["recon_decrease"] = 0.0

    # --- Issue #20: Hybridisation operator diagnostics ---

    # angular_displacement: how much Q moves on the S^{F-1} sphere per step.
    # Healthy: non-zero angular displacement during the active phase (H and
    # pump are both active, Q is moving on the sphere), decaying toward zero
    # near heat death (Q converges to a fixed point on the sphere).
    # A zero throughout means Q is stuck; a non-zero at heat death means
    # the system hasn't converged.
    if "angular_displacement" in metrics.columns:
        ang = metrics["angular_displacement"].to_numpy()
        ang_mean = float(np.mean(ang))
        # Score: full 5 pts if mean angular displacement is in [0.01, 0.5]
        # radians (~0.6 to ~28 degrees). Below 0.01: Q is stuck. Above 0.5:
        # Q is jittering, not converging.
        if ang_mean < 0.01:
            scores["angular_displacement"] = max(0.0, 5.0 * ang_mean / 0.01)
        elif ang_mean > 0.5:
            scores["angular_displacement"] = max(0.0, 5.0 * (1.0 - (ang_mean - 0.5) / 0.5))
        else:
            scores["angular_displacement"] = 5.0
    else:
        scores["angular_displacement"] = 0.0

    # pi_contraction_ratio: Pi operator activity.
    # Healthy range [0.01, 0.3] per step.
    if "pi_contraction_ratio" in metrics.columns:
        pcr = metrics["pi_contraction_ratio"].to_numpy()
        pcr_mean = float(np.mean(pcr))
        # Healthy range [0.3, 3.5] for S^{F-1}-normalised Q (||Q||=1).
        # IQR-calibrated: Q25=0.54, Q75=2.51, IQR=1.97, range=[0.3, 3.5].
        if pcr_mean < 0.3:
            scores["pi_contraction_ratio"] = max(0.0, 5.0 * pcr_mean / 0.3)
        elif pcr_mean > 3.5:
            scores["pi_contraction_ratio"] = max(0.0, 5.0 * (1.0 - (pcr_mean - 3.5) / 3.5))
        else:
            scores["pi_contraction_ratio"] = 5.0
    else:
        scores["pi_contraction_ratio"] = 0.0

    total = sum(scores.values())
    return {
        "total": round(total, 4),
        "criteria": scores,
        "S_0": S_0,
        "tau_range": tau_range,
        "alpha_bar_drop": ab_drop,
        "conserved_ratio": conserved_ratio,
        "heat_death_fired": hd > 0,
    }


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: compute real-data fitness from a CSV metrics file."""
    parser = argparse.ArgumentParser(
        prog="esdm-fitness",
        description="Compute ESDM real-data fitness score from metrics CSV.",
    )
    parser.add_argument("metrics_csv", type=str, help="Path to metrics.csv")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    metrics = pd.read_csv(args.metrics_csv)
    result = compute_fitness_real(metrics)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f'{result["total"]:.2f}')


if __name__ == "__main__":
    main()
