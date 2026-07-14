r"""ESDM real-world data loaders."""
from .cifar10 import CIFAR10DINOLoader, DEFAULT_OUTPUT_DIM
from .mnist import MNISTLoader, DEFAULT_PCA_DIM

__all__ = ["CIFAR10DINOLoader", "DEFAULT_OUTPUT_DIM", "MNISTLoader", "DEFAULT_PCA_DIM"]
