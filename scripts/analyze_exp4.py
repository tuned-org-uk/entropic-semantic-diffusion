"""
Analyze Experiment 4 results.

Reads ``runs/exp4/all_runs.csv`` (or a custom path) and produces:
  * per-variant MSE-vs-step summary statistics
  * Spearman rho(tau_k, explained_variance_k) table
  * 10-NN latent accuracy table
  * per-run Path A diagnostic summary
  * printed verdict against the issue #15 success criteria

Outputs are written to ``runs/exp4/analysis/`` by default.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUCCESS_THRESHOLDS = {
    "per_mode_mse_not_worse": True,  # ESDM-per-mode test MSE <= ESDM-global
    "tau_var_negative_corr": -0.3,
    "latent_knn_acc": 0.85,
}


def load_all_runs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Lists stored as strings in CSV; keep them as strings for final-row reads.
    for col in ("final_tau_k", "explained_variance"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def summarise_mse(df: pd.DataFrame) -> pd.DataFrame:
    """Final test MSE per variant aggregated over seeds."""
    last = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"]).tail(1)
    summary = (
        last.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale"])
        .agg(test_mse_mean=("test_mse", "mean"), test_mse_std=("test_mse", "std"))
        .reset_index()
    )
    summary["use_per_mode"] = summary["use_per_mode"].map({0: "global", 1: "per-mode"})
    return summary


def compare_global_vs_per_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Compare per-mode vs global test MSE per (kl_S_weight, a_tau_scale) pair."""
    last = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"]).tail(1)
    pivot = last.pivot_table(
        index=["kl_S_weight", "a_tau_scale", "seed"],
        columns="use_per_mode",
        values="test_mse",
    ).reset_index()
    pivot.columns = ["kl_S_weight", "a_tau_scale", "seed", "global_mse", "per_mode_mse"]
    pivot["per_mode_better_or_equal"] = pivot["per_mode_mse"] <= pivot["global_mse"]
    return pivot


def summarise_knn_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"]).tail(1)
    summary = (
        last.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale"])
        .agg(knn_mean=("latent_knn_acc", "mean"), knn_std=("latent_knn_acc", "std"))
        .reset_index()
    )
    summary["use_per_mode"] = summary["use_per_mode"].map({0: "global", 1: "per-mode"})
    return summary


def summarise_tau_variance_correlation(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"]).tail(1)
    summary = (
        last.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale"])
        .agg(corr_mean=("final_tau_var_corr", "mean"), corr_std=("final_tau_var_corr", "std"))
        .reset_index()
    )
    summary["use_per_mode"] = summary["use_per_mode"].map({0: "global", 1: "per-mode"})
    return summary


def summarise_path_a(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Path A diagnostics over training steps per run."""
    if "r_g_corr" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"])
    summary = grouped.agg(
        r_g_corr_mean=("r_g_corr", "mean"),
        r_g_corr_max=("r_g_corr", "max"),
        g_residual_mean=("g_residual_mean", "mean"),
        a_tau_final=("a_tau", "last"),
    ).reset_index()
    summary["use_per_mode"] = summary["use_per_mode"].map({0: "global", 1: "per-mode"})
    return summary


def build_verdict(df: pd.DataFrame) -> dict[str, dict]:
    """Compare aggregate results against issue #15 success criteria."""
    last = df.groupby(["kl_S_weight", "use_per_mode", "a_tau_scale", "seed"]).tail(1)
    verdict: dict[str, dict] = {}

    # 1. Per-mode test MSE not worse than global, averaged per condition.
    cmp = compare_global_vs_per_mode(df)
    if not cmp.empty:
        verdict["per_mode_mse_not_worse"] = {
            "value": float(cmp["per_mode_better_or_equal"].mean()),
            "threshold": ">= 0.5 (majority of seeds)",
            "pass": bool(cmp["per_mode_better_or_equal"].mean() >= 0.5),
        }

    # 2. Spearman rho(tau_k, explained_variance) < -0.3 for per-mode runs.
    per_mode = last[last["use_per_mode"] == 1]
    if not per_mode.empty and "final_tau_var_corr" in per_mode.columns:
        corr = float(per_mode["final_tau_var_corr"].mean())
        verdict["tau_var_negative_corr"] = {
            "value": corr,
            "threshold": "< -0.3",
            "pass": bool(corr < -0.3),
        }

    # 3. 10-NN latent accuracy >= 0.85.
    if not last.empty and "latent_knn_acc" in last.columns:
        knn = float(last["latent_knn_acc"].mean())
        verdict["latent_knn_acc"] = {
            "value": knn,
            "threshold": ">= 0.85",
            "pass": bool(knn >= 0.85),
        }

    return verdict


def main(csv_path: Path, output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    df = load_all_runs(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    mse_summary = summarise_mse(df)
    cmp_summary = compare_global_vs_per_mode(df)
    knn_summary = summarise_knn_accuracy(df)
    corr_summary = summarise_tau_variance_correlation(df)
    path_a_summary = summarise_path_a(df)
    verdict = build_verdict(df)

    mse_summary.to_csv(output_dir / "mse_summary.csv", index=False)
    cmp_summary.to_csv(output_dir / "global_vs_per_mode.csv", index=False)
    knn_summary.to_csv(output_dir / "knn_summary.csv", index=False)
    corr_summary.to_csv(output_dir / "tau_variance_corr_summary.csv", index=False)
    if not path_a_summary.empty:
        path_a_summary.to_csv(output_dir / "path_a_summary.csv", index=False)
    with open(output_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print("=" * 60)
    print("Experiment 4 Analysis")
    print("=" * 60)
    print("\n--- Final test MSE per variant ---")
    print(mse_summary.to_string(index=False))
    print("\n--- Global vs per-mode MSE comparison ---")
    print(cmp_summary.to_string(index=False))
    print("\n--- Latent 10-NN accuracy ---")
    print(knn_summary.to_string(index=False))
    print("\n--- Spearman rho(tau_k, explained_variance) ---")
    print(corr_summary.to_string(index=False))
    if not path_a_summary.empty:
        print("\n--- Path A diagnostics ---")
        print(path_a_summary.to_string(index=False))
    print("\n--- Verdict vs issue #15 success criteria ---")
    for name, v in verdict.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  {status}: {name} = {v['value']:.4f} (threshold {v['threshold']})")
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Experiment 4 sweep results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("runs/exp4/all_runs.csv"),
        help="Path to all_runs.csv (default: runs/exp4/all_runs.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/exp4/analysis"),
        help="Directory for analysis outputs (default: runs/exp4/analysis).",
    )
    args = parser.parse_args()
    main(args.csv, args.output_dir)
