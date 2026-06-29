"""
Phase 1 — Item-space fused matrix M_N.

M_N = alpha * norm(V @ V.T) - beta * norm(V @ L_F @ V.T)

where V ∈ R^{N×F} is the corpus embedding matrix and L_F ∈ R^{F×F} is the
feature-space graph Laplacian produced by the ArrowSpace GW algorithm.

The Rust ``build_laplacian_matrix`` receives the data matrix **already
transposed** (F×N) so its internal k-NN runs over feature-rows.  The
resulting L_F is therefore correctly oriented as F×F before it reaches this
module — no additional transpose is required here.

See docs/00-DESIGN.md §6 and docs/01-IMPLEMENTATION-PLAN.md §2 for the full
specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch import Tensor

# scipy sparse matrices are accepted from pyarrowspace's .to_csr() output
try:
    import scipy.sparse as sp  # type: ignore[import]
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


@dataclass
class DualSpaceMatrix:
    """
    Builds and caches the fused item-space geometric-semantic matrix M_N.

    M_N = alpha * norm(V @ V.T) - beta * norm(V @ L_F @ V.T)

    Parameters
    ----------
    alpha : float
        Weight on the pure geometric (cosine) term.  Default 0.5.
    beta  : float
        Weight on the spectral-semantic (Laplacian-modulated) term.  Default 0.5.

    Notes
    -----
    ``L_F`` is the **feature-space** Laplacian (F×F) produced by
    ``pyarrowspace.GraphLaplacian``.  The Rust core already transposes the
    data matrix to F×N before computing k-NN, so L_F arrives here in the
    correct orientation — rows and columns both index features.
    """

    alpha: float = 0.5
    beta: float = 0.5

    # Cached attributes (populated by build())
    M: Optional[Tensor] = field(default=None, repr=False)
    eigvals: Optional[Tensor] = field(default=None, repr=False)
    eigvecs: Optional[Tensor] = field(default=None, repr=False)
    fiedler_vec: Optional[Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #

    def build(
        self,
        V: Tensor,
        L_F: Union[Tensor, "sp.spmatrix"],  # noqa: F821
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Compute and cache M_N from corpus embeddings V and feature-space
        Laplacian L_F.

        Parameters
        ----------
        V : Tensor, shape (N, F)
            Corpus embedding matrix — N items, each with F features.
        L_F : Tensor or scipy sparse matrix, shape (F, F)
            Feature-space graph Laplacian from ``pyarrowspace.GraphLaplacian``.
            Accepts the dense Tensor from ``.to_dense()`` or the scipy sparse
            matrix from ``.to_csr()``.
        device : torch.device, optional
            Target device.  Defaults to V.device.
        """
        if device is None:
            device = V.device

        V = V.to(device=device, dtype=torch.float64)
        L = self._to_dense(L_F, device=device)  # (F, F)

        N, F = V.shape
        if L.shape != (F, F):
            raise ValueError(
                f"L_F shape {tuple(L.shape)} does not match V feature dim F={F}. "
                "Ensure L_F is the F×F feature-space Laplacian (not transposed)."
            )

        # Geometric term:  G = V @ V.T  →  (N, N)
        G = V @ V.T                          # (N, N)

        # Semantic term:   S = V @ L_F @ V.T  →  (N, N)
        # Equivalent to (V @ L_F) @ V.T; L_F is symmetric so no transpose needed
        S = (V @ L) @ V.T                    # (N, N)

        # Frobenius-normalise each term before mixing
        M = self.alpha * self._frobenius_norm(G) - self.beta * self._frobenius_norm(S)

        # Enforce exact symmetry (absorbs floating-point asymmetry)
        self.M = self._symmetrise(M)

        # Eigen-decomposition — eigh exploits symmetry, returns ascending order
        self.eigvals, self.eigvecs = torch.linalg.eigh(self.M)  # (N,), (N, N)

        # Fiedler vector: smallest non-trivial eigenvector
        # Skip the constant eigenvector (index 0 for connected graphs)
        self.fiedler_vec = self.eigvecs[:, 1]

    # ------------------------------------------------------------------ #

    def project(self, x: Tensor, k: int) -> Tensor:
        """
        Project item vectors x onto the top-k eigenvectors of M_N.

        Parameters
        ----------
        x : Tensor, shape (N, D) or (D,)
            Item vectors to project.
        k : int
            Number of eigenvectors to retain (largest eigenvalues).

        Returns
        -------
        Tensor, shape (N, k)
        """
        if self.eigvecs is None:
            raise RuntimeError("Call build() before project().")
        if k < 1 or k > self.eigvecs.shape[1]:
            raise ValueError(
                f"k={k} must be in [1, {self.eigvecs.shape[1]}]."
            )

        x = x.to(dtype=self.eigvecs.dtype, device=self.eigvecs.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)

        # Top-k eigenvectors are the last k columns (eigh returns ascending order)
        top_k_vecs = self.eigvecs[:, -k:]  # (N, k)

        # Project: x @ top_k_vecs  →  (batch, k)
        return x @ top_k_vecs

    # ------------------------------------------------------------------ #

    def rayleigh(self, x: Tensor) -> Tensor:
        """
        Compute Rayleigh quotient R(x) = x^T M_N x / x^T x for each row of x.

        Parameters
        ----------
        x : Tensor, shape (N, D) or (D,)

        Returns
        -------
        Tensor — scalar if x is 1-D, shape (N,) if x is 2-D.
        """
        if self.M is None:
            raise RuntimeError("Call build() before rayleigh().")

        squeeze = x.dim() == 1
        x = x.to(dtype=self.M.dtype, device=self.M.device)
        if squeeze:
            x = x.unsqueeze(0)  # (1, D)

        # Mx = x @ M  (works for batched rows)
        Mx = x @ self.M                          # (batch, N)
        numerator = (x * Mx).sum(dim=-1)         # (batch,)
        denominator = (x * x).sum(dim=-1).clamp(min=1e-12)

        result = numerator / denominator
        return result.squeeze(0) if squeeze else result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _frobenius_norm(A: Tensor) -> Tensor:
        """Frobenius-normalise a matrix; guarded against zero-norm."""
        return A / (A.norm("fro") + 1e-12)

    @staticmethod
    def _symmetrise(A: Tensor) -> Tensor:
        """Return (A + A.T) / 2 — enforces exact symmetry for eigh."""
        return (A + A.T) * 0.5

    @staticmethod
    def _to_dense(
        L: Union[Tensor, "sp.spmatrix"],  # noqa: F821
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """
        Convert L_F to a dense float64 Tensor on *device*.

        Accepts:
        - ``torch.Tensor``          — returned as-is (moved to device)
        - ``scipy.sparse`` matrix   — converted via ``.toarray()``
        - ``numpy.ndarray``         — wrapped directly
        """
        if isinstance(L, Tensor):
            return L.to(device=device, dtype=torch.float64)

        if _SCIPY_AVAILABLE and isinstance(L, sp.spmatrix):
            return torch.tensor(
                L.toarray(), dtype=torch.float64, device=device
            )

        # Fallback: assume numpy-array-like
        import numpy as np  # noqa: PLC0415
        return torch.tensor(
            np.asarray(L), dtype=torch.float64, device=device
        )
