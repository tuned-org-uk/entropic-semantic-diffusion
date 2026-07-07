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
import sys
from typing import Optional

import numpy as np
import pandas as pd

from esdm.experiments.exp3 import run_single


def compute_fitness(metrics: pd.DataFrame, bright_modes: int = 32) -> dict:
    """
    Compute the ESDM composite fitness score from exp3 metrics.

    Returns a dict with the total score (0-100) and per-criterion breakdown.
    """
    scores = {}
    n = len(metrics)
    if n == 0:
        return {"total": 0.0, "criteria": {}}

    # 1. tau_n monotonicity (10 pts)
    tau = metrics["tau_n"].to_numpy()
    is_monotone = np.all(np.diff(tau) >= -1e-12)
    tau_range = float(tau[-1] - tau[0])
    scores["tau_monotone"] = 10.0 if is_monotone else 0.0
    scores["tau_range_bonus"] = min(5.0, tau_range * 50.0)  # bonus for larger range

    # 2. alpha_bar decrease (10 pts) -- should drop from ~1 to ~0
    ab = metrics["alpha_bar"].to_numpy()
    ab_drop = float(ab[0] - ab[-1])
    scores["alpha_bar_decrease"] = min(10.0, ab_drop * 20.0)  # 0.5 drop -> 10 pts

    # 3. entropy conservation (15 pts) -- total entropy should stay ~constant
    # exp3 logs 'conserved' = kl_tau + dark_entropy (total, NOT residual).
    # The conservation law says |total - total[0]| should -> 0.
    total_entropy = metrics["conserved"].to_numpy()
    S_0 = float(total_entropy[0]) if len(total_entropy) > 0 else 1.0
    residual = np.abs(total_entropy - S_0)
    conserved_ratio = float(residual.mean()) / (abs(S_0) + 1e-8)
    scores["conservation"] = max(0.0, 15.0 * (1.0 - min(1.0, conserved_ratio)))

    # 4. heat death fires (10 pts)
    hd = int(metrics["is_heat_death"].sum())
    scores["heat_death"] = 10.0 if hd > 0 else 0.0
    # bonus: fires in the right window (not too early, not too late)
    if hd > 0:
        hd_step = int(metrics[metrics["is_heat_death"] == 1]["step"].iloc[0])
        ideal_frac = 0.7  # heat death should fire around 70% through
        frac = hd_step / n
        scores["heat_death_timing"] = max(0.0, 5.0 * (1.0 - abs(frac - ideal_frac) * 2))

    # 5. g_mean selectivity (10 pts) -- should be in [0.4, 0.8]
    if "g_mean" in metrics.columns:
        gm = float(metrics["g_mean"].mean())
        if 0.4 <= gm <= 0.8:
            scores["g_mean_selective"] = 10.0
        elif 0.1 <= gm <= 0.99:
            scores["g_mean_selective"] = 5.0  # partial credit
        else:
            scores["g_mean_selective"] = 0.0
    else:
        scores["g_mean_selective"] = 0.0

    # 6. g_residual_mean decrease (10 pts) -- should decrease toward 0.05
    if "g_residual_mean" in metrics.columns:
        grm = metrics["g_residual_mean"].to_numpy()
        grm_drop = float(grm[0] - grm[-1])
        scores["g_residual_decrease"] = max(0.0, min(10.0, grm_drop * 20.0))
        # bonus: final value near 0.05
        final_grm = float(grm[-1])
        scores["g_residual_final_bonus"] = max(0.0, 5.0 * (1.0 - abs(final_grm - 0.05) * 10))

    # 7. a_tau monotone decrease (5 pts)
    if "a_tau" in metrics.columns:
        at = metrics["a_tau"].to_numpy()
        at_monotone = np.all(np.diff(at) <= 1e-6)
        scores["a_tau_monotone"] = 5.0 if at_monotone else 0.0
        # bonus: reaches near 0
        scores["a_tau_final_bonus"] = max(0.0, 3.0 * (1.0 - float(at[-1]) / (float(at[0]) + 1e-8)))

    # 8. r_g_corr positive and increasing (10 pts)
    if "r_g_corr" in metrics.columns:
        rgc = metrics["r_g_corr"].to_numpy()
        rgc_mean = float(rgc.mean())
        rgc_final = float(rgc[-1])
        if rgc_final > 0:
            scores["r_g_corr_positive"] = min(10.0, rgc_final * 10.0)
        elif rgc_mean > 0:
            scores["r_g_corr_positive"] = 3.0  # partial credit
        else:
            scores["r_g_corr_positive"] = 0.0
        # bonus: increasing trend
        if n > 10:
            first_half = float(rgc[:n // 2].mean())
            second_half = float(rgc[n // 2:].mean())
            if second_half > first_half:
                scores["r_g_corr_increasing"] = 5.0

    # 9. rho_sign stabilization (10 pts) -- variance should decrease over time
    rs = metrics["rho_sign"].to_numpy()
    if n > 20:
        first_var = float(np.var(rs[:n // 3]))
        last_var = float(np.var(rs[-n // 3:]))
        if last_var < first_var:
            scores["rho_sign_stabilizing"] = 10.0
        else:
            scores["rho_sign_stabilizing"] = 0.0
    else:
        scores["rho_sign_stabilizing"] = 0.0

    # 10. dark_entropy floor in [35%, 85%] of S_0 (10 pts)
    if "dark_entropy_floor" in metrics.columns:
        floor = float(metrics["dark_entropy_floor"].iloc[-1])
        floor_ratio = floor / (abs(S_0) + 1e-8)
        if 0.35 <= floor_ratio <= 0.85:
            scores["dark_entropy_floor"] = 10.0
        elif 0.2 <= floor_ratio <= 0.95:
            scores["dark_entropy_floor"] = 5.0
        else:
            scores["dark_entropy_floor"] = 0.0
    else:
        scores["dark_entropy_floor"] = 0.0

    total = sum(scores.values())
    return {"total": round(total, 4), "criteria": scores, "S_0": S_0,
            "tau_range": tau_range, "alpha_bar_drop": ab_drop,
            "conserved_ratio": conserved_ratio, "heat_death_fired": hd > 0}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-fitness",
        description="Compute ESDM composite fitness score from exp3 run.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--kl-s-weight", type=float, default=0.05)
    parser.add_argument("--kl-decay", type=float, default=0.03)
    parser.add_argument("--lambda-clip", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--a-tau-scale", type=float, default=5.0)
    parser.add_argument("--bright-modes", type=int, default=32)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    metrics = run_single(
        n_steps=args.n_steps,
        kl_S_weight=args.kl_s_weight,
        kl_decay=args.kl_decay,
        lambda_clip=args.lambda_clip,
        rho=args.rho,
        a_tau_scale=args.a_tau_scale,
        bright_modes=args.bright_modes,
        seed=args.seed,
    )

    result = compute_fitness(metrics, bright_modes=args.bright_modes)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"ESDM Fitness Score: {result['total']:.2f} / 100")
        print(f"  S_0={result['S_0']:.4f}  tau_range={result['tau_range']:.4f}  "
              f"alpha_bar_drop={result['alpha_bar_drop']:.4f}  "
              f"conserved_ratio={result['conserved_ratio']:.4f}  "
              f"heat_death={result['heat_death_fired']}")
        print("  Criteria breakdown:")
        for k, v in sorted(result["criteria"].items()):
            print(f"    {k:30s} {v:6.2f}")


if __name__ == "__main__":
    main()
