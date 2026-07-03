"""
Experiment 3 tests — Barrier Height Sweep on High-Dimensional Gaussian Clusters.

Tests the consolidated spec from issue #6 (2026-07-03):
  * dim=64, bright_modes=32 (= dim // 2)
  * Path A bivariate-gate diagnostics
  * a_tau_scale sweep at the primary condition
"""
import numpy as np
import pandas as pd
import pytest
import torch

from esdm.experiments.exp3 import (
    DIM,
    BRIGHT_MODES,
    make_gaussian_clusters,
    run_single,
    run_sweep,
    assert_success_criteria,
)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_make_gaussian_clusters_shape_and_dim():
    X = make_gaussian_clusters(dim=DIM, n_clusters=4, samples_per_cluster=100, rho=3.0, seed=0)
    assert X.shape == (400, DIM)
    assert X.dtype == torch.float64


def test_make_gaussian_clusters_has_cluster_structure():
    X = make_gaussian_clusters(dim=DIM, n_clusters=4, samples_per_cluster=250, rho=10.0, seed=0)
    # With high SNR the batch-mean should be close to the origin (centres cancel).
    assert X.mean(dim=0).abs().max() < 2.0


def test_bright_modes_is_half_dim():
    assert BRIGHT_MODES == DIM // 2
    assert BRIGHT_MODES == 32
    assert DIM == 64


# ---------------------------------------------------------------------------
# Single-run structural tests (fast)
# ---------------------------------------------------------------------------

def test_run_single_returns_expected_columns():
    metrics = run_single(n_steps=50, kl_S_weight=0.05, kl_decay=0.03, seed=0)
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 50

    expected = {
        "step",
        "seed",
        "kl_decay",
        "clock_eps",
        "lambda_clip",
        "bright_modes",
        "a_tau_scale",
        "kl_tau",
        "dark_entropy",
        "conserved",
        "Q_t_0",
        "rho_plus",
        "rho_minus",
        "rho_sign",
        "dt",
        "dt_cfl_bound",
        "delta_tau",
        "tau_n",
        "norm_tau",
        "Lambda_tau",
        "alpha_bar",
        "is_heat_death",
        "r_k_mean",
        "r_k_max",
        "g_residual_mean",
        "a_tau",
        "r_g_corr",
        "g_mean",
        "g_entropy",
        "lambda_cap_step",
        "rho_sign_crossing_step",
        "dark_entropy_floor",
        "dim",
        "kl_S_weight",
        "rho_snr",
    }
    assert expected.issubset(set(metrics.columns))


def test_run_single_default_geometry():
    metrics = run_single(n_steps=10, seed=0)
    assert metrics["dim"].iloc[0] == DIM
    assert metrics["bright_modes"].iloc[0] == BRIGHT_MODES
    assert metrics["a_tau_scale"].iloc[0] == 5.0


def test_run_single_no_nan_or_inf():
    metrics = run_single(n_steps=50, seed=0)
    numeric = metrics.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_run_single_tau_n_increases_monotonically():
    metrics = run_single(n_steps=50, seed=0)
    tau = metrics["tau_n"].to_numpy()
    assert np.all(np.diff(tau) >= -1e-12)


def test_run_single_alpha_bar_decreases():
    metrics = run_single(n_steps=50, seed=0)
    assert metrics["alpha_bar"].iloc[-1] < metrics["alpha_bar"].iloc[0]


def test_run_single_norm_tau_in_unit_interval():
    metrics = run_single(n_steps=50, seed=0)
    assert metrics["norm_tau"].between(0.0, 1.0).all()


def test_run_single_a_tau_monotone_decreasing():
    metrics = run_single(n_steps=50, seed=0)
    a_tau = metrics["a_tau"].to_numpy()
    assert np.all(np.diff(a_tau) <= 1e-6)


def test_path_a_columns_present_and_bounded():
    metrics = run_single(n_steps=50, seed=0)
    assert metrics["r_k_mean"].ge(0.0).all()
    assert metrics["r_k_max"].ge(0.0).all()
    assert metrics["g_residual_mean"].between(0.0, 1.0).all()
    # a_tau starts positive and decays to zero.
    assert metrics["a_tau"].iloc[0] > 0.0
    assert metrics["a_tau"].iloc[-1] <= metrics["a_tau"].iloc[0]


def test_run_single_a_tau_scale_2_uses_different_gate():
    m5 = run_single(n_steps=50, a_tau_scale=5.0, seed=0)
    m2 = run_single(n_steps=50, a_tau_scale=2.0, seed=0)
    # a_tau starts close to a_tau_scale (slightly below because tau_n > 0
    # after the first clock step has been logged).
    assert m5["a_tau"].iloc[0] > 4.5
    assert m2["a_tau"].iloc[0] > 1.8
    assert m5["a_tau"].iloc[0] > m2["a_tau"].iloc[0]
    # Higher authority (larger a_tau_scale) pushes the clock-driven offset
    # b(tau) harder, so the mean gate should differ from the lower authority.
    assert not np.isclose(
        m5["g_residual_mean"].mean(), m2["g_residual_mean"].mean(), atol=1e-3
    )


# ---------------------------------------------------------------------------
# Sweep structural tests
# ---------------------------------------------------------------------------

def test_sweep_grid_includes_a_tau_scale_at_primary_condition_only():
    # Minimal 2-step sweep at primary condition only.
    df = run_sweep(
        kl_s_weights=[0.05],
        kl_decays=[0.03],
        rho_snrs=[3.0],
        lambda_clips=[10.0],
        seeds=[0, 1],
        a_tau_scales=[2.0, 5.0],
        n_steps=10,
        output_root="runs/exp3_test_sweep",
    )
    assert len(df) == 2 * 2 * 10  # 2 seeds * 2 a_tau_scale * 10 steps
    assert set(df["a_tau_scale"].unique()) == {2.0, 5.0}


def test_sweep_row_count_matches_cartesian_product():
    df = run_sweep(
        kl_s_weights=[0.001, 0.01],
        kl_decays=[0.03],
        rho_snrs=[3.0],
        lambda_clips=[10.0],
        seeds=[0],
        a_tau_scales=[5.0],
        n_steps=20,
        output_root="runs/exp3_test_sweep2",
    )
    assert len(df) == 2 * 1 * 1 * 1 * 1 * 1 * 20


# ---------------------------------------------------------------------------
# Success-criteria tests
# ---------------------------------------------------------------------------

def test_assert_success_criteria_passes_for_valid_run():
    # Long enough to satisfy dark-entropy floor and rho_sign crossing.
    metrics = run_single(n_steps=300, kl_S_weight=0.05, kl_decay=0.03, seed=0)
    assert_success_criteria(metrics, kl_S_weight=0.05, kl_decay=0.03, lambda_clip=10.0)


def test_assert_success_criteria_fails_for_zero_dark_entropy():
    metrics = run_single(n_steps=50, kl_S_weight=0.05, seed=0)
    metrics["dark_entropy"] = 0.0
    with pytest.raises(AssertionError, match="dark_entropy"):
        assert_success_criteria(metrics, kl_S_weight=0.05)


def test_assert_success_criteria_detects_conservation_violation():
    metrics = run_single(n_steps=50, kl_S_weight=0.05, seed=0)
    metrics["conserved"] = metrics["conserved"] + 1.0
    with pytest.raises(AssertionError, match="conservation"):
        assert_success_criteria(metrics, kl_S_weight=0.05)


# ---------------------------------------------------------------------------
# Regime-specific tests (moderate length)
# ---------------------------------------------------------------------------

def test_low_barrier_shows_oscillations():
    # tau_n is monotone by construction; the cyclic signature appears in
    # rho_sign crossing below 1.0 (constructive-dominant regime).
    metrics = run_single(n_steps=500, kl_S_weight=0.001, kl_decay=0.03, seed=0)
    n_crossings = int((metrics["rho_sign"] < 1.0).sum())
    assert n_crossings >= 10, f"Expected many constructive-dominant steps, got {n_crossings}"


def test_high_barrier_reaches_heat_death():
    # Use a faster decay so heat death fires within the test window.
    metrics = run_single(n_steps=600, kl_S_weight=1.0, kl_decay=0.08, seed=0)
    hd = metrics[metrics["is_heat_death"] == 1]
    assert not hd.empty, "Expected heat death to fire"
    assert int(hd["step"].iloc[0]) < 500, "Expected heat death before step 500"


def test_dark_entropy_floor_in_valid_range():
    metrics = run_single(n_steps=300, kl_S_weight=0.05, kl_decay=0.03, seed=0)
    S_0 = float(metrics["conserved"].iloc[0]) or 1.0
    floor = float(metrics["dark_entropy_floor"].iloc[-1])
    assert 0.40 * S_0 <= floor <= 0.70 * S_0, (
        f"dark_entropy floor {floor:.4f} outside [40%, 70%] of S_0={S_0:.4f}"
    )
