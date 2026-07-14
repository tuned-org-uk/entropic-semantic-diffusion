"""Fast smoke tests for Experiment 4 (MNIST) — issue #22."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from esdm.experiments import exp4


class _FakeMNISTLoader:
    """Deterministic fake loader that avoids MNIST download in fast tests."""

    dim_input = 16

    def __init__(self, *args, **kwargs) -> None:
        rng = np.random.default_rng(3407)
        self._train = torch.from_numpy(
            self._l2_normalise(rng.standard_normal((256, 16))).astype(np.float64)
        )
        self._test = torch.from_numpy(
            self._l2_normalise(rng.standard_normal((64, 16))).astype(np.float64)
        )

    @staticmethod
    def _l2_normalise(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
        return x / norms

    def train_batch(self, batch_size: int, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self._train.shape[0], size=batch_size)
        return self._train[idx]

    def test_batch(self, batch_size: int, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self._test.shape[0], size=batch_size)
        return self._test[idx]


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(
        "esdm.cli.train.MNISTLoader", _FakeMNISTLoader, raising=False
    )
    return _FakeMNISTLoader


def test_exp4_run_single_smoke(fake_loader, monkeypatch, tmp_path):
    """Exp4 run_single should complete a few steps without error."""
    metrics = exp4.run_single(
        n_steps=3,
        batch_size=8,
        wht_filter="none",
        seed=3407,
        output_dir=tmp_path,
    )
    assert len(metrics) == 3
    assert "loss" in metrics.columns
    assert "tau_n" in metrics.columns


def test_exp4_main_cli_exists():
    """main_cli entry point should be importable and callable."""
    assert callable(exp4.main_cli)


@pytest.mark.slow
def test_exp4_real_mnist_short(tmp_path):
    """Slow: run a handful of steps on the real MNIST PCA-16 loader."""
    metrics = exp4.run_single(
        n_steps=3,
        batch_size=8,
        wht_filter="none",
        seed=3407,
        output_dir=tmp_path,
        data_dir=".pytest_cache/mnist",
    )
    assert len(metrics) == 3
