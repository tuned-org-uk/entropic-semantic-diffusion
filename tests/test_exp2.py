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
