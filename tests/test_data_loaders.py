"""Tests for ESDM real-world data loaders (issue #21)."""
from __future__ import annotations

import pytest
import torch

from esdm.data.mnist import MNISTLoader, DEFAULT_PCA_DIM


class TestMNISTLoader:
    """Unit tests for the MNIST PCA-16 loader."""

    @pytest.fixture(scope="class")
    def loader(self):
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
