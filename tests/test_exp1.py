"""
Phase 1 experiment test — Entropic Clock calibration (Experiment 1).
"""
from esdm.experiments.exp1 import (
    assert_success_criteria,
    build_two_community_laplacian,
    run_bridge_sweep,
    run_experiment,
)


def test_build_two_community_laplacian_shape():
    L = build_two_community_laplacian(n_per_community=5, bridge_weight=0.1)
    assert L.shape == (10, 10)
    assert (L == L.T).all()


def test_run_experiment_passes_criteria():
    metrics, clock = run_experiment(bridge_weight=0.1, n_steps=250, seed=3407)
    assert len(metrics) == 250
    assert_success_criteria(metrics, clock)


def test_run_bridge_sweep():
    results = run_bridge_sweep(
        bridge_weights=(0.0, 0.01, 0.1, 1.0),
        seed=3407,
    )
    assert set(results.keys()) == {0.0, 0.01, 0.1, 1.0}
    for metrics in results.values():
        assert len(metrics) == 250
