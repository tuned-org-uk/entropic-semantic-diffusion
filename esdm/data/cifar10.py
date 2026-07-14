r"""
CIFAR-10 DINO-256 data loader for ESDM.

Produces 256-D embeddings of CIFAR-10 images by:
1. Downloading CIFAR-10 directly (no torchvision).
2. Resizing 32x32 RGB images to 224x224 with SciPy.
3. Encoding them with a frozen DINO-small ViT (``facebook/dino-vits16``).
4. Projecting the 384-D CLS token to 256-D via PCA.
5. L2-normalising each embedding.

The loader is lazy: downloads and model loading happen on first access.
"""
from __future__ import annotations

import logging
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy import ndimage
from torch import Tensor
from transformers import AutoModel

logger = logging.getLogger(__name__)

#: CIFAR-10 official download URL.
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#: ImageNet mean/std used by DINO-s pretraining.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64).reshape(1, 3, 1, 1)

#: DINO-small model name.
_DINO_MODEL = "facebook/dino-vits16"

#: Default output embedding dimension.
DEFAULT_OUTPUT_DIM = 256


class CIFAR10DINOLoader:
    """
    Lazily-downloaded CIFAR-10 train/test loader emitting DINO-256 embeddings.

    Parameters
    ----------
    output_dim : int
        Target embedding dimension after PCA projection (default 256).
    data_dir : Path or str, optional
        Directory where downloaded/cached data is stored. Defaults to
        ``./data/cifar10``.
    seed : int
        Random seed used for subset selection and batch sampling.
    pca_fit_samples : int
        Number of train images used to fit the PCA projection.
    dino_batch_size : int
        Batch size used when running the frozen DINO encoder.
    dino_model_name : str
        Hugging Face model name for the frozen DINO encoder.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        data_dir: Optional[Path | str] = None,
        seed: int = 3407,
        pca_fit_samples: int = 5000,
        dino_batch_size: int = 64,
        dino_model_name: str = _DINO_MODEL,
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
    ) -> None:
        self.output_dim = output_dim
        self.data_dir = Path(data_dir or "data/cifar10")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.pca_fit_samples = pca_fit_samples
        self.dino_batch_size = dino_batch_size
        self.dino_model_name = dino_model_name
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples

        self._train: Optional[Tensor] = None
        self._test: Optional[Tensor] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._dino: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------ #

    def _download(self) -> None:
        """Download and extract CIFAR-10 if not already present."""
        extracted = self.data_dir / "cifar-10-batches-py"
        expected_files = {f"data_batch_{i}" for i in range(1, 6)} | {"test_batch", "batches.meta"}
        if extracted.exists() and expected_files.issubset({p.name for p in extracted.iterdir()}):
            return
        archive = self.data_dir / "cifar-10-python.tar.gz"
        expected_bytes = 170498071
        if not archive.exists() or archive.stat().st_size != expected_bytes:
            logger.info("Downloading CIFAR-10 from %s ...", _CIFAR10_URL)
            tmp = self.data_dir / "cifar-10-python.tar.gz.tmp"
            urllib.request.urlretrieve(_CIFAR10_URL, tmp)
            size = tmp.stat().st_size
            if size != expected_bytes:
                raise RuntimeError(
                    f"CIFAR-10 download incomplete: {size} bytes (expected {expected_bytes})"
                )
            tmp.replace(archive)
        logger.info("Extracting CIFAR-10 archive ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(self.data_dir, filter="data")

    @staticmethod
    def _unpickle(path: Path) -> dict:
        with path.open("rb") as f:
            return pickle.load(f, encoding="bytes")

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw CIFAR-10 images and labels."""
        self._download()
        root = self.data_dir / "cifar-10-batches-py"

        train_images: list[np.ndarray] = []
        train_labels: list[int] = []
        for i in range(1, 6):
            batch = self._unpickle(root / f"data_batch_{i}")
            train_images.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])
        train_x = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32).astype(np.float64)
        train_y = np.asarray(train_labels, dtype=np.int64)

        test_batch = self._unpickle(root / "test_batch")
        test_x = test_batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float64)
        test_y = np.asarray(test_batch[b"labels"], dtype=np.int64)
        return train_x, train_y, test_x, test_y

    @staticmethod
    def _resize(images: np.ndarray, size: int = 224) -> np.ndarray:
        """Resize a batch of CHW images using SciPy (bilinear-ish)."""
        if images.shape[2:] == (size, size):
            return images
        zoom = (1, 1, size / images.shape[2], size / images.shape[3])
        return ndimage.zoom(images, zoom, order=1, mode="nearest")

    @staticmethod
    def _normalise(images: np.ndarray) -> np.ndarray:
        """Scale to [0, 1] then apply ImageNet mean/std."""
        images = images / 255.0
        return (images - _IMAGENET_MEAN) / _IMAGENET_STD

    def _load_dino(self) -> torch.nn.Module:
        """Lazy-load the frozen DINO encoder."""
        if self._dino is None:
            logger.info("Loading frozen DINO encoder %s ...", self.dino_model_name)
            self._dino = AutoModel.from_pretrained(self.dino_model_name)
            self._dino.eval()
            for p in self._dino.parameters():
                p.requires_grad = False
        return self._dino

    def _dino_embed(self, images: np.ndarray) -> np.ndarray:
        """Encode a batch of preprocessed images to CLS embeddings."""
        model = self._load_dino()
        device = next(model.parameters()).device
        embeddings: list[np.ndarray] = []
        n = images.shape[0]
        with torch.no_grad():
            for i in range(0, n, self.dino_batch_size):
                batch = torch.from_numpy(images[i : i + self.dino_batch_size]).float()
                batch = batch.to(device)
                out = model(batch)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls)
        return np.concatenate(embeddings, axis=0)

    @staticmethod
    def _l2_normalise(x: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation."""
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
        return x / norms

    def _fit(self) -> None:
        """Download, encode, project, and cache train/test tensors."""
        if self._train is not None:
            return

        train_x, train_y, test_x, test_y = self._load_raw()

        rng = np.random.default_rng(self.seed)
        pca_idx = rng.choice(train_x.shape[0], size=min(self.pca_fit_samples, train_x.shape[0]), replace=False)

        logger.info("Resizing CIFAR-10 images for DINO ...")
        train_resized = self._resize(train_x[pca_idx])
        train_resized = self._normalise(train_resized)
        logger.info("Encoding PCA-fit subset (%d images) ...", train_resized.shape[0])
        train_embed_pca = self._dino_embed(train_resized)

        self._pca = PCA(n_components=self.output_dim, random_state=self.seed)
        self._pca.fit(train_embed_pca)
        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        logger.info(
            "CIFAR-10 DINO PCA-%d fitted. Variance retained: %.4f",
            self.output_dim,
            float(self._explained_variance_ratio.sum()),
        )

        logger.info("Encoding full train split ...")
        train_x = train_x[: self.max_train_samples]
        train_y = train_y[: self.max_train_samples]
        train_resized_full = self._resize(train_x)
        train_resized_full = self._normalise(train_resized_full)
        train_embed_full = self._dino_embed(train_resized_full)
        train_proj = self._pca.transform(train_embed_full)

        logger.info("Encoding test split ...")
        test_x = test_x[: self.max_test_samples]
        test_y = test_y[: self.max_test_samples]
        test_resized = self._resize(test_x)
        test_resized = self._normalise(test_resized)
        test_embed = self._dino_embed(test_resized)
        test_proj = self._pca.transform(test_embed)

        self._train = torch.from_numpy(self._l2_normalise(train_proj).astype(np.float64))
        self._test = torch.from_numpy(self._l2_normalise(test_proj).astype(np.float64))
        self._train_labels = train_y
        self._test_labels = test_y

    # ------------------------------------------------------------------ #

    @property
    def dim_input(self) -> int:
        """Embedding dimension $F$ (equals ``output_dim``)."""
        return self.output_dim

    @property
    def train_data(self) -> Tensor:
        """Full train split, shape ``(50000, output_dim)``."""
        self._fit()
        return self._train  # type: ignore[return-value]

    @property
    def test_data(self) -> Tensor:
        """Full test split, shape ``(10000, output_dim)``."""
        self._fit()
        return self._test  # type: ignore[return-value]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component explained variance fractions of the DINO->target projection."""
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
