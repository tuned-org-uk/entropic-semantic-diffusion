"""Checks on the synthetic evidence protocol, independent of model quality."""

import numpy as np

from esdm.experiments import v2_readout as v2


def test_path_data_has_analytic_modes_and_independent_holdout():
    data = v2.path_data()
    k = np.arange(v2.FEATURES)
    eigenvalues = 2 - 2 * np.cos(np.pi * k / v2.FEATURES)
    np.testing.assert_allclose(data.laplacian @ data.basis, data.basis * eigenvalues, atol=1e-12)
    np.testing.assert_allclose(data.basis.T @ data.basis, np.eye(v2.FEATURES), atol=1e-12)
    assert not np.array_equal(data.train.clean, data.test.clean)
    assert not np.array_equal(data.train.observed - data.train.clean,
                              data.test.observed - data.test.clean)


def test_paired_interval_resamples_cases_together():
    case_effect = np.linspace(4, 500, 31)
    contrast = v2.paired_contrast(case_effect - 0.25, case_effect)
    np.testing.assert_allclose([contrast["mean"], *contrast["ci95"]], -0.25)


def test_training_batches_use_each_training_case_once_per_epoch():
    batches = v2.training_batches()
    assert batches.shape == (32, 32)
    epochs = batches.reshape(2, v2.N_TRAIN)
    for epoch in epochs:
        np.testing.assert_array_equal(np.sort(epoch), np.arange(v2.N_TRAIN))
    assert not np.array_equal(epochs[0], epochs[1])


def test_readout_controls_preserve_normalised_offsets_and_radius():
    from esdm.models.bounded_readout import BoundedSpectralReadout

    data = v2.path_data()
    result = v2.decoding_experiment(data)
    arms = result["arms"]
    radius = np.asarray(result["radius"])
    prime = np.asarray(arms["prime"]["initial"]["gains"])
    shuffled = np.asarray(arms["shuffled"]["initial"]["gains"])
    permutation = result["shuffle_permutation"]
    np.testing.assert_allclose(((prime - 1) / radius)[permutation], (shuffled - 1) / radius)
    assert not np.array_equal(prime, shuffled)
    np.testing.assert_array_equal(arms["identity"]["initial"]["gains"],
                                  arms["smooth"]["initial"]["gains"])
    np.testing.assert_allclose(radius, BoundedSpectralReadout(v2.RANK).radius.detach().numpy())
    for name in ("smooth", "prime", "shuffled"):
        assert np.all(np.abs(np.asarray(arms[name]["trained"]["gains"]) - 1) <= radius + 1e-12)
