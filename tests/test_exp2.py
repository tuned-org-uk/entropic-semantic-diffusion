"""
Phase 2 experiment test — Vibrational Wave Recurrence on a 1-D Harmonic Chain.
"""
import numpy as np
import torch

from esdm.experiments.exp2 import (
    assert_success_criteria,
    fiedler_vector,
    path_graph_eigenvalues,
    path_graph_laplacian,
    run_experiment,
)


def test_path_graph_laplacian_eigenvalues():
    n = 32
    L = path_graph_laplacian(n)
    assert L.shape == (n, n)
    assert torch.allclose(L, L.T)

    analytic = path_graph_eigenvalues(n)
    computed = torch.linalg.eigvalsh(L).numpy()
    assert np.allclose(np.sort(analytic), np.sort(computed), atol=1e-10)


def test_fiedler_vector_normalised():
    n = 32
    v = fiedler_vector(n)
    assert abs(v.norm().item() - 1.0) < 1e-10


def test_run_experiment_passes_criteria():
    metrics = run_experiment(n=32, n_steps=500, seed=3407)
    assert len(metrics) == 500
    assert_success_criteria(metrics)

def test_logged_columns_match_shared_convention():
    metrics = run_experiment(n=32, n_steps=500, seed=3407)
    expected = {
        "step",
        "seed",
        "kl_decay",
        "clock_eps",
        "lambda_clip",
        "bright_modes",
        "kl_tau",
        "dark_entropy",
        "g_mean",
        "g_entropy",
        "conserved",
        "Q_t_0",
        "rho_plus",
        "rho_minus",
        "rho_sign",
        "dt",
        "dt_cfl_bound",
        "delta_tau",
        "tau_n",
        "Lambda_tau",
        "alpha_bar",
        "is_heat_death",
    }
    assert expected.issubset(set(metrics.columns))
    assert metrics["dark_entropy"].ge(0.0).all()
    assert metrics["g_entropy"].ge(0.0).all()
    assert metrics["rho_sign"].between(0.0, 2.0).all()
    # Default lambda_clip should be the lowered value.
    assert metrics["lambda_clip"].iloc[0] == 15.0


def test_dark_entropy_proxy_nonzero_after_initial_transient():
    """The spectral dark-entropy proxy should become positive once energy
    spreads beyond the initial Fiedler mode."""
    metrics = run_experiment(n=32, n_steps=500, seed=3407)
    late = metrics[metrics["step"] >= 100]["dark_entropy"]
    assert (late > 0.0).any(), "dark_entropy proxy stayed zero after transient"

