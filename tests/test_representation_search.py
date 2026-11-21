"""Probe fitting must not use held-out targets."""

import numpy as np
import pytest
import torch

from esdm.experiments.representation_search import fit_probe, probe_errors, confirmation_accepts
from esdm.experiments import representation_search as search
from esdm.experiments.v2_readout import _model, path_data, _rng


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


def test_search_restores_saved_lazy_state_before_evaluation(tmp_path):
    data = path_data()
    model = _model(data, 8)
    x = torch.from_numpy(data.train.clean[:4]).float()
    model(x)
    path = tmp_path / "winner.pt"
    search.checkpoint(model, path, {"arm": "chart"})
    restored = search.restore_checked(model, path)
    for a, b in zip(model.encode(x, generator=torch.Generator().manual_seed(3407)),
                    restored.encode(x, generator=torch.Generator().manual_seed(3407))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_failed_trial_preserves_completed_update_count():
    data = path_data()
    # Corrupt one case in batch two; the first update must still be accounted for.
    index = _rng(102).permutation(512)[64]
    data.train.clean[index] = float("nan")
    with pytest.raises(search.TrialFailure) as failure:
        search.train(data, "chart", 0.0003, 16, {})
    assert failure.value.record["updates"] == 1
    assert failure.value.record["failure_step"] == 2
    assert len(failure.value.record["trace"]) == 1
