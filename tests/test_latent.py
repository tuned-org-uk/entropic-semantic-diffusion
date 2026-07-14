"""Tests for latent-space evaluation helpers (issue #22)."""
from __future__ import annotations

import pytest
import torch

from esdm.eval.latent import (
    conservation_drift,
    evaluate_latent,
    knn_latent_accuracy,
    reconstruction_mse,
)


class _TinyModel:
    """Minimal model stand-in for latent-eval tests."""

    def __init__(self, dim: int = 8, latent: int = 4) -> None:
        self.enc_mu = torch.nn.Linear(dim, latent, dtype=torch.float64)
        self.S_0 = torch.tensor(1.0, dtype=torch.float64)
        self.training = True

    def __call__(self, x: torch.Tensor):
        class Out:
            x_hat = x
        return Out()

    def health_check(self):
        return {"conserved": 1.0}

    def eval(self):
        pass

    def train(self):
        pass


def test_reconstruction_mse():
    x = torch.ones(4, 8, dtype=torch.float64)
    x_hat = torch.zeros(4, 8, dtype=torch.float64)
    assert reconstruction_mse(x, x_hat) == pytest.approx(1.0, abs=1e-6)


def test_conservation_drift():
    model = _TinyModel()
    x = torch.randn(4, 8, dtype=torch.float64)
    assert conservation_drift(model, x) == pytest.approx(0.0, abs=1e-6)


def test_knn_latent_accuracy():
    model = _TinyModel()
    x_train = torch.eye(8, dtype=torch.float64)
    y_train = torch.arange(8)
    x_test = torch.eye(8, dtype=torch.float64)
    y_test = torch.arange(8)
    acc = knn_latent_accuracy(model, x_train, y_train, x_test, y_test, k=1)
    assert 0.0 <= acc <= 1.0


def test_evaluate_latent_with_labels():
    model = _TinyModel()

    class _FakeLoader:
        _test_targets = torch.arange(10)

        def test_batch(self, batch_size: int, seed: int = 0):
            return torch.randn(batch_size, 8, dtype=torch.float64)

        def train_batch(self, batch_size: int, seed: int = 0):
            return torch.randn(batch_size, 8, dtype=torch.float64)

        @property
        def test_data(self):
            return torch.randn(10, 8, dtype=torch.float64)

    loader = _FakeLoader()
    results = evaluate_latent(model, loader, n_test=8, k=1)
    assert "reconstruction_mse" in results
    assert "conservation_drift" in results
    assert "knn_accuracy" in results
