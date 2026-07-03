"""
Experiment 4 smoke tests — fast structural checks for the MNIST-PCA pipeline.

These tests deliberately use very short training runs (50-200 steps) to keep
CI fast.  They verify that:
  * MNIST loads and projects to the requested PCA dimension.
  * ESDM-global and ESDM-per-mode both train without NaN/Inf.
  * The per-mode variant accumulates a non-trivial per_mode_tau vector.
  * Success-criteria assertions can be evaluated.
"""
import ast
import numpy as np
import pandas as pd
import pytest
import torch

from esdm.experiments.exp4 import (
    PCA_DIM,
    load_mnist_pca,
    make_esdm_cfg,
    run_single,
    run_sweep,
    assert_success_criteria,
    reconstruction_mse,
    latent_knn_accuracy,
)
from esdm.model import ESDM


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def _mnist_available() -> bool:
    """Probe whether MNIST can be loaded without network access."""
    try:
        from sklearn.datasets import fetch_openml
        fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_load_mnist_pca_shape():
    X_train, X_test, y_train, y_test, labels_train, labels_test, pca, _ = load_mnist_pca(dim=PCA_DIM)
    assert X_train.shape == (60000, PCA_DIM)
    assert X_test.shape == (10000, PCA_DIM)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    assert labels_train.shape == (60000,)
    assert labels_test.shape == (10000,)
    assert pca.n_components_ == PCA_DIM


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_load_mnist_pca_unit_norm():
    X_train, X_test, _, _, _, _, _, _ = load_mnist_pca(dim=PCA_DIM)
    train_norms = torch.linalg.norm(X_train, dim=1)
    test_norms = torch.linalg.norm(X_test, dim=1)
    assert torch.allclose(train_norms, torch.ones_like(train_norms), atol=1e-6)
    assert torch.allclose(test_norms, torch.ones_like(test_norms), atol=1e-6)


# ---------------------------------------------------------------------------
# Model builder tests
# ---------------------------------------------------------------------------

def test_make_esdm_cfg_per_mode_flag():
    cfg_scalar = make_esdm_cfg(dim=16, use_per_mode=False)
    cfg_per = make_esdm_cfg(dim=16, use_per_mode=True)
    assert cfg_scalar["model"]["use_per_mode"] is False
    assert cfg_per["model"]["use_per_mode"] is True


def test_esdm_per_mode_instantiates():
    cfg = make_esdm_cfg(dim=16, use_per_mode=True)
    model = ESDM(cfg)
    assert model.use_per_mode is True
    assert model.diffusion_decoder.use_per_mode is True


# ---------------------------------------------------------------------------
# Training-loop structural tests (fast, require MNIST)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_run_single_global_returns_expected_columns():
    metrics = run_single(
        dim=16,
        use_per_mode=False,
        max_steps=50,
        eval_every=25,
        seed=0,
    )
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 50
    expected = {
        "step", "seed", "dim", "use_per_mode", "kl_S_weight",
        "train_loss", "train_recon", "kl_S", "kl_tau", "dark_entropy",
        "conserved", "tau_n", "alpha_bar", "Lambda_tau", "rho_sign",
        "test_mse", "train_mse", "latent_knn_acc",
        "best_test_mse", "final_tau_var_corr",
    }
    assert expected.issubset(set(metrics.columns))


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_run_single_per_mode_accumulates_tau_vector():
    metrics = run_single(
        dim=16,
        use_per_mode=True,
        max_steps=50,
        eval_every=25,
        seed=0,
    )
    tau_k = metrics.iloc[-1]["final_tau_k"]
    if isinstance(tau_k, str):
        tau_k = ast.literal_eval(tau_k)
    tau_k = np.asarray(tau_k, dtype=float)
    assert tau_k.shape == (16,)
    assert np.all(tau_k > 0), "per_mode_tau must be strictly positive"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_run_single_no_nan_or_inf():
    metrics = run_single(
        dim=16,
        use_per_mode=True,
        max_steps=50,
        eval_every=25,
        seed=0,
    )
    numeric = metrics.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_run_single_tau_n_increases():
    metrics = run_single(
        dim=16,
        use_per_mode=False,
        max_steps=50,
        eval_every=25,
        seed=0,
    )
    tau = metrics["tau_n"].to_numpy()
    assert np.all(np.diff(tau) >= -1e-12)


# ---------------------------------------------------------------------------
# Evaluator tests (require MNIST)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_reconstruction_mse_finite():
    cfg = make_esdm_cfg(dim=16, use_per_mode=False)
    model = ESDM(cfg)
    X_train, X_test, _, _, _, _, _, _ = load_mnist_pca(dim=16)
    mse = reconstruction_mse(model, X_test[:100])
    assert np.isfinite(mse)
    assert mse >= 0.0


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_latent_knn_accuracy_runs():
    cfg = make_esdm_cfg(dim=16, use_per_mode=False)
    model = ESDM(cfg)
    X_train, X_test, y_train, y_test, _, _, _, _ = load_mnist_pca(dim=16)
    acc = latent_knn_accuracy(
        model,
        X_train[:500], y_train[:500],
        X_test[:100], y_test[:100],
        n_neighbors=5,
        max_train=500,
    )
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Success-criteria tests (require MNIST)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_assert_success_criteria_passes_for_valid_short_run():
    metrics = run_single(
        dim=16,
        use_per_mode=False,
        max_steps=100,
        eval_every=50,
        seed=0,
    )
    verdict = assert_success_criteria(metrics)
    # Short runs may not satisfy all thresholds; only hard checks are required.
    assert verdict["finite_training"] is True
    assert verdict["test_mse_finite"] is True
    assert verdict["all_tau_positive"] is True


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_assert_success_criteria_detects_nan():
    metrics = run_single(
        dim=16,
        use_per_mode=False,
        max_steps=50,
        eval_every=25,
        seed=0,
    )
    metrics.loc[0, "train_loss"] = np.nan
    verdict = assert_success_criteria(metrics)
    assert verdict["finite_training"] is False


# ---------------------------------------------------------------------------
# Sweep structural tests (require MNIST)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(not _mnist_available(), reason="MNIST not available (no network/cache)")
def test_run_sweep_limited_grid():
    df = run_sweep(
        dims=[16],
        kl_s_weights=[0.01],
        seeds=[0],
        use_per_modes=[False, True],
        max_steps=20,
        output_root="runs/exp4_test_sweep",
    )
    assert len(df) == 2 * 20  # 2 variants * 20 steps
    assert set(df["use_per_mode"].unique()) == {0, 1}
