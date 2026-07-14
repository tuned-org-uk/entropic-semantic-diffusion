"""Tests for ESDM real-world data loaders (issue #21).

Test strategy:
- Fast unit tests (default): CIFAR-10 helper functions and loader API surface.
- Medium tests: MNIST loader (one OpenML download + PCA fit, cached per session).
- Slow tests: CIFAR-10 full loader (frozen DINO encoding), run with --run-slow.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from esdm.data.cifar10 import CIFAR10DINOLoader, DEFAULT_OUTPUT_DIM
from esdm.data.mnist import MNISTLoader, DEFAULT_PCA_DIM


# ---------------------------------------------------------------------------
# MNIST loader tests (medium: one download + PCA fit)
# ---------------------------------------------------------------------------


class TestMNISTLoader:
    """Critical tests for the MNIST PCA-16 loader."""

    @classmethod
    @pytest.fixture(scope="class")
    def loader(cls):
        # Cache inside pytest_cache so repeated runs are fast after first fit.
        return MNISTLoader(pca_dim=DEFAULT_PCA_DIM, data_dir=".pytest_cache/mnist")

    def test_train_and_test_shapes_and_dtype(self, loader):
        train = loader.train_data
        test = loader.test_data
        assert train.shape == (60000, DEFAULT_PCA_DIM)
        assert test.shape == (10000, DEFAULT_PCA_DIM)
        assert train.dtype == torch.float64
        assert test.dtype == torch.float64

    def test_batches_are_l2_normalised_and_correct_shape(self, loader):
        batch = loader.train_batch(batch_size=32, seed=3407)
        assert batch.shape == (32, DEFAULT_PCA_DIM)
        norms = batch.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_loader_is_deterministic(self, loader):
        b1 = loader.train_batch(batch_size=16, seed=123)
        b2 = loader.train_batch(batch_size=16, seed=123)
        assert torch.allclose(b1, b2)


# ---------------------------------------------------------------------------
# CIFAR-10 loader tests (slow: DINO download/encode)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCIFAR10DINOLoader:
    """Critical end-to-end tests for the CIFAR-10 DINO-256 loader."""

    @classmethod
    @pytest.fixture(scope="class")
    def loader(cls):
        return CIFAR10DINOLoader(
            output_dim=DEFAULT_OUTPUT_DIM,
            data_dir=".pytest_cache/cifar10",
            pca_fit_samples=100,
            dino_batch_size=16,
            max_train_samples=200,
            max_test_samples=50,
        )

    def test_train_and_test_shapes_and_dtype(self, loader):
        train = loader.train_data
        test = loader.test_data
        assert train.shape == (200, DEFAULT_OUTPUT_DIM)
        assert test.shape == (50, DEFAULT_OUTPUT_DIM)
        assert train.dtype == torch.float64
        assert test.dtype == torch.float64

    def test_batches_are_l2_normalised_and_correct_shape(self, loader):
        batch = loader.train_batch(batch_size=16, seed=3407)
        assert batch.shape == (16, DEFAULT_OUTPUT_DIM)
        norms = batch.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_loader_is_deterministic(self, loader):
        b1 = loader.train_batch(batch_size=16, seed=123)
        b2 = loader.train_batch(batch_size=16, seed=123)
        assert torch.allclose(b1, b2)


# ---------------------------------------------------------------------------
# CIFAR-10 helper tests (fast: pure numpy, no model/download)
# ---------------------------------------------------------------------------


class TestCIFAR10Helpers:
    """Fast unit tests for CIFAR-10 preprocessing helpers."""

    def test_resize_preserves_batch_and_channels(self):
        images = np.zeros((4, 3, 32, 32), dtype=np.float64)
        resized = CIFAR10DINOLoader._resize(images, size=224)
        assert resized.shape == (4, 3, 224, 224)

    def test_normalise_uses_imagenet_stats(self):
        images = np.ones((2, 3, 224, 224), dtype=np.float64) * 255.0
        normed = CIFAR10DINOLoader._normalise(images)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        assert np.allclose(normed, (1.0 - mean) / std, atol=1e-6)

    def test_l2_normalise(self):
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
        normed = CIFAR10DINOLoader._l2_normalise(x)
        assert np.allclose(normed[0], np.array([0.6, 0.8]))
        assert np.allclose(normed[1], np.array([0.0, 1.0]))

    def test_loader_dim_input(self):
        loader = CIFAR10DINOLoader(output_dim=256, data_dir=".pytest_cache/cifar10")
        assert loader.dim_input == 256
