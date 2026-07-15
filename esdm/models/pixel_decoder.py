"""
Pixel decoder for ESDM generation (issue #23).

Inverts a D-dimensional embedding back to image pixels. Used after the
reverse-SDE sampler produces embeddings — this maps them to visible images.

Architecture: small convolutional MLP that maps dim_input -> H*W*C.
Trained jointly with the ESDM or separately on (embedding, image) pairs.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PixelDecoder(nn.Module):
    """
    Maps an embedding vector to an image.

    Parameters
    ----------
    dim_input : int
        Embedding dimension (e.g. 256 for DINO-PCA, 16 for MNIST-PCA).
    img_channels : int
        Number of image channels (3 for RGB, 1 for grayscale).
    img_size : int
        Square image side length (32 for CIFAR-10, 28 for MNIST).
    hidden_dim : int
        Internal MLP width.
    """

    def __init__(
        self,
        dim_input: int,
        img_channels: int = 3,
        img_size: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.img_channels = img_channels
        self.img_size = img_size

        # MLP that expands embedding to a flat image, then a small conv
        # refines it.
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, img_channels * img_size * img_size),
        )

        # Light 2-layer conv refinement.
        self.refine = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, img_channels, 3, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, dim_input)
            Embedding to decode.

        Returns
        -------
        Tensor (B, C, H, W) in [0, 1].
        """
        B = z.shape[0]
        flat = self.mlp(z)
        img = flat.view(B, self.img_channels, self.img_size, self.img_size)
        img = self.refine(img)
        return img
