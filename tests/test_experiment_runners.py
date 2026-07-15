"""
Experiment runner integration tests.

Exercises the main_cli / run_bridge_sweep entry points as pytest tests.
Only configurations that complete in under ~5s per test are included here.

Ref: docs/legacy/02-EXPERIMENTS.md, issue #6
"""
import pandas as pd

from esdm.experiments.exp1 import run_bridge_sweep
from esdm.experiments.exp2 import main_cli as exp2_main_cli


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

    results = run_bridge_sweep(
        bridge_weights=(0.0, 0.01, 0.1, 1.0),
        base_output_dir=str(tmp_path),
        seed=3407,
    )
    for w, metrics in results.items():
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
    assert metrics["lambda_clip"].iloc[0] == 20.0


def test_exp2_main_cli_writes_output_dir(tmp_path):
    """main_cli creates the output directory and writes metrics CSV."""
    out = tmp_path / "exp2_out"
    exp2_main_cli(output_dir=str(out), seed=3407, n_steps=500)
    assert out.exists()
    csv_files = list(out.glob("*.csv"))
    assert len(csv_files) >= 1
