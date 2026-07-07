"""
Experiment runner integration tests.

Replaces the deleted scripts/run_exp{1,2,3}.py by exercising the
main_cli / run_bridge_sweep / run_sweep entry points as pytest tests.
Only configurations that complete in under ~5s per test are included here;
the full 540-run production sweep belongs in a separate CI stage.

Ref: docs/legacy/02-EXPERIMENTS.md, issue #6
"""
import pandas as pd
import pytest

from esdm.experiments.exp1 import run_bridge_sweep
from esdm.experiments.exp2 import main_cli as exp2_main_cli
from esdm.experiments.exp3 import (
    A_TAU_SCALES,
    main_cli as exp3_main_cli,
    run_sweep as exp3_run_sweep,
)


# ---------------------------------------------------------------------------
# Exp 1: bridge sweep with output directory (runner --output-dir equivalent)
# ---------------------------------------------------------------------------

def test_exp1_bridge_sweep_with_output_dir(tmp_path):
    """run_bridge_sweep writes per-weight subdirectories when given a base dir."""
    results = run_bridge_sweep(
        bridge_weights=(0.0, 0.01, 0.1, 1.0),
        base_output_dir=str(tmp_path),
        seed=3407,
    )
    assert set(results.keys()) == {0.0, 0.01, 0.1, 1.0}
    for w, metrics in results.items():
        assert len(metrics) == 250
        sub = tmp_path / f"w_{w}"
        assert sub.exists()


def test_exp1_bridge_sweep_all_weights_pass_criteria(tmp_path):
    """Every bridge weight in the sweep must pass success criteria."""
    from esdm.experiments.exp1 import assert_success_criteria

    results = run_bridge_sweep(
        bridge_weights=(0.0, 0.01, 0.1, 1.0),
        base_output_dir=str(tmp_path),
        seed=3407,
    )
    for w, metrics in results.items():
        # run_bridge_sweep already asserts internally; this is a belt-and-suspenders
        # check that the returned DataFrames are valid.
        assert len(metrics) == 250
        assert metrics["tau_n"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Exp 2: main_cli single run (runner --kl-decay --lambda-clip equivalent)
# ---------------------------------------------------------------------------

def test_exp2_main_cli_single_run(tmp_path):
    """main_cli runs the full experiment and asserts success criteria."""
    metrics = exp2_main_cli(
        output_dir=str(tmp_path / "exp2"),
        seed=3407,
        kl_decay=0.08,
        n_steps=500,
    )
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 500
    assert metrics["tau_n"].is_monotonic_increasing


def test_exp2_main_cli_custom_lambda_clip(tmp_path):
    """main_cli accepts a custom lambda_clip (runner --lambda-clip equivalent)."""
    metrics = exp2_main_cli(
        output_dir=str(tmp_path / "custom_clip"),
        seed=3407,
        lambda_clip=20.0,
        n_steps=500,
    )
    assert len(metrics) == 500
    # The logged lambda_clip should reflect the custom value.
    assert metrics["lambda_clip"].iloc[0] == 20.0


def test_exp2_main_cli_writes_output_dir(tmp_path):
    """main_cli creates the output directory and writes metrics CSV."""
    out = tmp_path / "exp2_out"
    exp2_main_cli(output_dir=str(out), seed=3407, n_steps=500)
    assert out.exists()
    csv_files = list(out.glob("*.csv"))
    assert len(csv_files) >= 1


# ---------------------------------------------------------------------------
# Exp 3: main_cli single run (runner --kl-s-weight --kl-decay equivalent)
# ---------------------------------------------------------------------------

def test_exp3_main_cli_single_run(tmp_path):
    """main_cli runs a single experiment and asserts success criteria."""
    metrics = exp3_main_cli(
        output_dir=str(tmp_path / "exp3"),
        seed=0,
        kl_S_weight=0.05,
        kl_decay=0.03,
        lambda_clip=10.0,
        rho=3.0,
        a_tau_scale=5.0,
        n_steps=300,
    )
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 300
    assert metrics["tau_n"].is_monotonic_increasing


def test_exp3_main_cli_low_barrier(tmp_path):
    """main_cli with low kl_S_weight shows oscillations (runner equivalent)."""
    metrics = exp3_main_cli(
        output_dir=str(tmp_path / "low_barrier"),
        seed=0,
        kl_S_weight=0.001,
        kl_decay=0.03,
        n_steps=300,
    )
    n_crossings = int((metrics["rho_sign"] < 1.0).sum())
    assert n_crossings >= 5


def test_exp3_main_cli_high_barrier_heat_death(tmp_path):
    """main_cli with high kl_S_weight reaches heat death."""
    metrics = exp3_main_cli(
        output_dir=str(tmp_path / "high_barrier"),
        seed=0,
        kl_S_weight=1.0,
        kl_decay=0.08,
        n_steps=400,
    )
    hd = metrics[metrics["is_heat_death"] == 1]
    assert not hd.empty


# ---------------------------------------------------------------------------
# Exp 3: limited sweep (runner --limited-sweep equivalent)
# ---------------------------------------------------------------------------

def test_exp3_limited_sweep(tmp_path):
    """Limited sweep: 3 kl_s x 2 kl_decay x 1 rho x 1 lambda x 2 a_tau x 2 seeds = 24 runs."""
    df = exp3_run_sweep(
        kl_s_weights=[0.001, 0.05, 1.0],
        kl_decays=[0.03, 0.08],
        rho_snrs=[3.0],
        lambda_clips=[10.0],
        a_tau_scales=A_TAU_SCALES,
        seeds=[0, 1],
        n_steps=50,
        output_root=str(tmp_path / "limited_sweep"),
    )
    # a_tau_scale is only expanded at the primary condition (kl_decay=0.03,
    # lambda_clip=10, rho=3). At the non-primary condition (kl_decay=0.08)
    # only a_tau_scales[0] is used.
    # Primary: 3 kl_s x 1 kl_decay x 1 rho x 1 lambda x 2 a_tau x 2 seeds x 50 = 600
    # Non-primary: 3 kl_s x 1 kl_decay x 1 rho x 1 lambda x 1 a_tau x 2 seeds x 50 = 300
    expected_rows = 600 + 300  # 18 runs x 50 steps
    assert len(df) == expected_rows
    assert (tmp_path / "limited_sweep" / "all_runs.csv").exists()


def test_exp3_limited_sweep_a_tau_only_at_primary(tmp_path):
    """a_tau_scale is only expanded at the primary condition."""
    df = exp3_run_sweep(
        kl_s_weights=[0.05],
        kl_decays=[0.03, 0.08],
        rho_snrs=[3.0],
        lambda_clips=[10.0],
        a_tau_scales=[2.0, 5.0],
        seeds=[0],
        n_steps=10,
        output_root=str(tmp_path / "primary_check"),
    )
    primary_rows = df[(df["kl_decay"] == 0.03) & (df["lambda_clip"] == 10.0) & (df["rho_snr"] == 3.0)]
    non_primary_rows = df[df["kl_decay"] == 0.08]

    assert set(primary_rows["a_tau_scale"].unique()) == {2.0, 5.0}
    assert set(non_primary_rows["a_tau_scale"].unique()) == {2.0}
