"""Probe fitting must not use held-out targets."""

import numpy as np

from esdm.experiments.representation_search import fit_probe, probe_errors, confirmation_accepts


def test_probe_recovers_affine_signal_and_heldout_targets_only_affect_scoring():
    rng = np.random.default_rng(3407)
    train = rng.normal(size=(32, 3))
    test = rng.normal(size=(9, 3))
    weight = rng.normal(size=(3, 6))
    bias = rng.normal(size=6)
    decoder = fit_probe(train, train @ weight + bias)
    np.testing.assert_allclose(probe_errors(decoder, test, test @ weight + bias), 0, atol=1e-25)
    np.testing.assert_allclose(probe_errors(decoder, test, test @ weight + bias + 2), 4)


def test_confirmation_requires_both_controls_practical_gain_and_uncertainty():
    good = {"relative_improvement": 0.02, "ci95": [-0.02, -0.01]}
    assert confirmation_accepts([good, good])
    assert not confirmation_accepts([good, {**good, "relative_improvement": 0.005}])
    assert not confirmation_accepts([good, {**good, "ci95": [-0.02, 0.01]}])
    assert not confirmation_accepts([good, {**good, "relative_improvement": float("nan")}])
