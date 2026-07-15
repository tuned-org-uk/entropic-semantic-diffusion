"""
ESDM fitness evaluation -- composite health score from docs/00-DESIGN.md §8.1.

Runs exp3 run_single with given parameters, measures 10 health criteria,
returns a score in [0, 100]. Higher is better.

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

        Path-A diagnostics (which are NaN when the filter is disabled) and focuses
    on training health: entropic-clock motion, conservation, reconstruction
    progress, and heat-death behaviour.
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
        frac = hd_step / n
        scores["heat_death_timing"] = max(0.0, 5.0 * (1.0 - abs(frac - 0.7) * 2))

    # Rho-sign stabilisation.
    rs = metrics["rho_sign"].to_numpy()
    if n > 20:
        first_var = float(np.var(rs[: n // 3]))
        last_var = float(np.var(rs[-n // 3 :]))
        scores["rho_sign_stabilizing"] = 10.0 if last_var < first_var else 0.0

    # Reconstruction/ELBO progress.
    loss = metrics["loss"].to_numpy()
    loss_drop = float(loss[0] - loss[-1])
    scores["loss_decrease"] = (
        min(10.0, max(0.0, (loss_drop / (abs(loss[0]) + 1e-8)) * 10.0))
        if loss[0] != 0
        else 0.0
    )

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
