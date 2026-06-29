"""
Phase 1 — Item-space fused matrix M_N.

M_N = alpha * norm(V @ V.T) - beta * norm(V @ L_F @ V.T)

where V ∈ R^{N×F} is the corpus embedding matrix and L_F ∈ R^{F×F} is the
feature-space graph Laplacian produced by the ArrowSpace GW algorithm.

See docs/01-IMPLEMENTATION-PLAN.md §2 for the full specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class DualSpaceMatrix:
    """
    Builds and caches the fused item-space geometric-semantic matrix M_N.

    Parameters
    ----------
    alpha : float   weight on geometric term  (default 0.5)
    beta  : float   weight on semantic term   (default 0.5)
    """

    alpha: float = 0.5
    beta: float = 0.5

    # Cached attributes (populated by build())
    M: Optional[Tensor] = field(default=None, repr=False)
    eigvals: Optional[Tensor] = field(default=None, repr=False)
    eigvecs: Optional[Tensor] = field(default=None, repr=False)
    fiedler_vec: Optional[Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #

    def build(self, V: Tensor, L_F: Tensor) -> None:
        """
        Compute and cache M_N from corpus embeddings V (N, F) and
        feature-space Laplacian L_F (F, F).
        """
        raise NotImplementedError("Phase 1: implement DualSpaceMatrix.build()")

    # ------------------------------------------------------------------ #

    def project(self, x: Tensor, k: int) -> Tensor:
        """
        Project item vectors x (N, D) onto the top-k eigenvectors of M_N.
        Returns Tensor (N, k).
        """
        raise NotImplementedError("Phase 1: implement DualSpaceMatrix.project()")

    # ------------------------------------------------------------------ #

    def rayleigh(self, x: Tensor) -> Tensor:
        """
        Compute Rayleigh quotient R(x) = x^T M_N x / x^T x for each row of x.
        x shape: (N, D) or (D,). Returns scalar or (N,).
        """
        raise NotImplementedError("Phase 1: implement DualSpaceMatrix.rayleigh()")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _frobenius_norm(A: Tensor) -> Tensor:
        return A / (A.norm("fro") + 1e-12)
