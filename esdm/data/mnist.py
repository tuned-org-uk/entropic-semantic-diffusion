r"""
MNIST data loader for ESDM.

Produces 16-D PCA projections of MNIST digits, L2-normalised, as the input
embedding $x \in \mathbb{R}^{16}$ used by Experiment 4.

The implementation avoids torchvision; it downloads MNIST via
``sklearn.datasets.fetch_openml`` and fits PCA with scikit-learn.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from torch import Tensor

logger = logging.getLogger(__name__)

#: MNIST OpenML identifier for the 784-pixel version.
_OPENML_NAME = "mnist_784"

#: Default PCA target dimension for Experiment 4.
DEFAULT_PCA_DIM = 16


class MNISTLoader:
    """
    Lazily-downloaded MNIST train/test loader emitting PCA-16 embeddings.

    Parameters
    ----------
    pca_dim : int
        Number of principal components to retain (default 16).
    data_dir : Path or str, optional
        Directory where downloaded/cached data is stored. Defaults to
        ``./data/mnist``.
    seed : int
        Random seed used only for shuffling batches.
    """

    def __init__(
        self,
        pca_dim: int = DEFAULT_PCA_DIM,
        data_dir: Optional[Path | str] = None,
        seed: int = 3407,
    ) -> None:
        self.pca_dim = pca_dim
        self.data_dir = Path(data_dir or "data/mnist")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        self._train: Optional[Tensor] = None
        self._test: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None
        self._test_targets: Optional[Tensor] = None
        self._pca: Optional[PCA] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    def _download(self) -> tuple[np.ndarray, np.ndarray]:
        """Download MNIST 784 from OpenML and return (data, targets)."""
        logger.info("Downloading MNIST from OpenML (this may take a moment)...")
        bunch = fetch_openml(
            _OPENML_NAME,
            version=1,
            parser="auto",
            as_frame=False,
            cache=True,
            data_home=str(self.data_dir),
        )
        data = np.asarray(bunch.data, dtype=np.float64)
        targets = np.asarray(bunch.target, dtype=np.int64)
        return data, targets

    @staticmethod
    def _l2_normalise(x: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation."""
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
        return x / norms

    def _fit(self) -> None:
        """Download (if needed), fit PCA, and cache train/test tensors."""
        if self._train is not None:
            return

        data, targets = self._download()
        # OpenML ordering: first 60k are the canonical train split, last 10k test.
        train_x = data[:60000]
        test_x = data[60000:]
        train_y = targets[:60000]
        test_y = targets[60000:]

        self._pca = PCA(n_components=self.pca_dim, random_state=self.seed)
        train_proj = self._pca.fit_transform(train_x)
        test_proj = self._pca.transform(test_x)
        self._explained_variance_ratio = self._pca.explained_variance_ratio_

        logger.info(
            "MNIST PCA-%d fitted. Variance retained: %.4f",
            self.pca_dim,
            float(self._explained_variance_ratio.sum()),
        )

        self._train = torch.from_numpy(
            self._l2_normalise(train_proj).astype(np.float64)
        )
        self._test = torch.from_numpy(
            self._l2_normalise(test_proj).astype(np.float64)
        )
        self._train_targets = torch.from_numpy(train_y)
        self._test_targets = torch.from_numpy(test_y)

    # ------------------------------------------------------------------ #

    @property
    def dim_input(self) -> int:
        """Embedding dimension $F$ (equals ``pca_dim``)."""
        return self.pca_dim

    @property
    def train_data(self) -> Tensor:
        """Full train split, shape ``(60000, pca_dim)``."""
        self._fit()
        return self._train  # type: ignore[return-value]

    @property
    def test_data(self) -> Tensor:
        """Full test split, shape ``(10000, pca_dim)``."""
        self._fit()
        return self._test  # type: ignore[return-value]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component explained variance fractions."""
        self._fit()
        return self._explained_variance_ratio  # type: ignore[return-value]

    # ------------------------------------------------------------------ #

    def train_batch(self, batch_size: int, seed: Optional[int] = None) -> Tensor:
        """Random batch from the train split."""
        self._fit()
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        idx = rng.integers(0, self._train.shape[0], size=batch_size)
        return self._train[idx]

    def test_batch(self, batch_size: int, seed: Optional[int] = None) -> Tensor:
        """Random batch from the test split."""
        self._fit()
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        idx = rng.integers(0, self._test.shape[0], size=batch_size)
        return self._test[idx]
