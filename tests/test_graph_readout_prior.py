"""Ensure the readout receives a feature-space chart on the fallback path."""

import torch

from esdm.graphs import wiring


def test_fallback_features_and_items_have_distinct_node_axes(monkeypatch):
    monkeypatch.setattr(wiring, "_HAS_ASP", False)
    features = torch.randn(13, 5, generator=torch.Generator().manual_seed(3407))
    feature_prior = wiring.WireGraph.build(features, mode="feature", k=2)
    expected = wiring.WireGraph.build(features.T, mode="item", k=2)
    torch.testing.assert_close(feature_prior.laplacian_dense, expected.laplacian_dense)
    assert feature_prior.lambdas is None
