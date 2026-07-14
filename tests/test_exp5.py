"""Fast smoke tests for Experiment 5 (CIFAR-10) — issue #22."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from esdm.experiments import exp5


class _FakeCIFAR10Loader:
    """Deterministic fake loader that avoids CIFAR-10 download in fast tests."""

    dim_input = 256

    def __init__(self, *args, **kwargs) -> None:
        rng = np.random.default_rng(3407)
        self._train = torch.from_numpy(
            self._l2_normalise(rng.standard_normal((256, 256))).astype(np.float64)
        )
        self._test = torch.from_numpy(
            self._l2_normalise(rng.standard_normal((64, 256))).astype(np.float64)
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

    @property
    def train_data(self) -> torch.Tensor:
        return self._train

    @property
    def test_data(self) -> torch.Tensor:
        return self._test


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(
        "esdm.experiments.exp5.CIFAR10DINOLoader", _FakeCIFAR10Loader, raising=False
    )
    return _FakeCIFAR10Loader


def test_exp5_run_single_smoke(fake_loader, monkeypatch, tmp_path):
    """Exp5 run_single should complete a few steps without error."""
    metrics = exp5.run_single(
        n_steps=2,
        batch_size=8,
        wht_filter="none",
        seed=3407,
        output_dir=tmp_path,
    )
    assert len(metrics) == 2
    assert "loss" in metrics.columns
    assert "tau_n" in metrics.columns


def test_exp5_main_cli_exists():
    """main_cli entry point should be importable and callable."""
    assert callable(exp5.main_cli)


@pytest.mark.slow
def test_exp5_real_cifar10_short(tmp_path):
    """Slow: run a handful of steps on the real CIFAR-10 DINO-256 loader."""
    metrics = exp5.run_single(
        n_steps=2,
        batch_size=8,
        wht_filter="none",
        seed=3407,
        output_dir=tmp_path,
        data_dir=".pytest_cache/cifar10",
        max_train_samples=50,
        max_test_samples=10,
        pca_fit_samples=25,
    )
    assert len(metrics) == 2
