"""Tests for ESDM real-world data loaders (issue #21)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from esdm.data.cifar10 import CIFAR10DINOLoader, DEFAULT_OUTPUT_DIM
from esdm.data.mnist import MNISTLoader, DEFAULT_PCA_DIM


class TestMNISTLoader:
    """Unit tests for the MNIST PCA-16 loader."""

    @classmethod
    @pytest.fixture(scope="class")
    def loader(cls):
        # Use a temporary cache directory inside pytest_cache.
        return MNISTLoader(pca_dim=DEFAULT_PCA_DIM, data_dir=".pytest_cache/mnist")

    def test_dim_input_matches_pca_dim(self, loader):
        assert loader.dim_input == DEFAULT_PCA_DIM

    def test_train_data_shape(self, loader):
        train = loader.train_data
        assert train.shape == (60000, DEFAULT_PCA_DIM)
        assert train.dtype == torch.float64

    def test_test_data_shape(self, loader):
        test = loader.test_data
        assert test.shape == (10000, DEFAULT_PCA_DIM)
        assert test.dtype == torch.float64

    def test_l2_normalised(self, loader):
        train = loader.train_data
        norms = train.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_batch_shape(self, loader):
        batch = loader.train_batch(batch_size=32, seed=3407)
        assert batch.shape == (32, DEFAULT_PCA_DIM)

    def test_variance_retained(self, loader):
        ratio = loader.explained_variance_ratio
        assert ratio.shape == (DEFAULT_PCA_DIM,)
        assert float(ratio.sum()) > 0.55  # MNIST PCA-16 retains ~60% variance on raw pixels

    def test_deterministic_with_seed(self, loader):
        b1 = loader.train_batch(batch_size=16, seed=123)
        b2 = loader.train_batch(batch_size=16, seed=123)
        assert torch.allclose(b1, b2)



@pytest.mark.slow
class TestCIFAR10DINOLoader:
    """Unit tests for the CIFAR-10 DINO-256 loader (slow; requires DINO download/encode)."""

    @classmethod
    @pytest.fixture(scope="class")
    def loader(cls):
        # Use a tiny subset so the frozen DINO encoder runs in test time.
        return CIFAR10DINOLoader(
            output_dim=DEFAULT_OUTPUT_DIM,
            data_dir=".pytest_cache/cifar10",
            pca_fit_samples=100,
            dino_batch_size=16,
            max_train_samples=200,
            max_test_samples=50,
        )

    def test_dim_input_matches_output_dim(self, loader):
        assert loader.dim_input == DEFAULT_OUTPUT_DIM

    def test_train_data_shape(self, loader):
        train = loader.train_data
        assert train.shape[0] == 200
        assert train.shape[1] == DEFAULT_OUTPUT_DIM
        assert train.dtype == torch.float64

    def test_test_data_shape(self, loader):
        test = loader.test_data
        assert test.shape[0] == 50
        assert test.shape[1] == DEFAULT_OUTPUT_DIM
        assert test.dtype == torch.float64

    def test_l2_normalised(self, loader):
        train = loader.train_data
        norms = train.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_batch_shape(self, loader):
        batch = loader.train_batch(batch_size=16, seed=3407)
        assert batch.shape == (16, DEFAULT_OUTPUT_DIM)

    def test_explained_variance_shape(self, loader):
        ratio = loader.explained_variance_ratio
        assert ratio.shape == (DEFAULT_OUTPUT_DIM,)
        assert float(ratio.sum()) > 0.0

    def test_deterministic_with_seed(self, loader):
        b1 = loader.train_batch(batch_size=16, seed=123)
        b2 = loader.train_batch(batch_size=16, seed=123)
        assert torch.allclose(b1, b2)



class TestCIFAR10Helpers:
    """Fast unit tests for CIFAR-10 loader helpers (no DINO encoding)."""

    def test_resize_shape(self):
        images = np.zeros((4, 3, 32, 32), dtype=np.float64)
        resized = CIFAR10DINOLoader._resize(images, size=224)
        assert resized.shape == (4, 3, 224, 224)

    def test_normalise_matches_imagenet(self):
        images = np.ones((2, 3, 224, 224), dtype=np.float64) * 255.0
        normed = CIFAR10DINOLoader._normalise(images)
        # At max input (255), normalised value is (1 - mean) / std for each channel.
        expected = (1.0 - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        assert np.allclose(normed, expected, atol=1e-6)

    def test_l2_normalise(self):
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
        normed = CIFAR10DINOLoader._l2_normalise(x)
        assert np.allclose(normed[0], np.array([0.6, 0.8]))
        assert np.allclose(normed[1], np.array([0.0, 1.0]))

    def test_loader_dim_input(self):
        loader = CIFAR10DINOLoader(output_dim=256, data_dir=".pytest_cache/cifar10")
        assert loader.dim_input == 256
