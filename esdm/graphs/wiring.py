"""
ArrowSpace adapter — the single surface through which the 2.5-D track
accesses graph construction, the sparse Laplacian, the Rayleigh/λτ score,
and the spectral item-coordinate embedding.

Wraps the real `arrowspace==0.26.2` Rust bindings. Primitives missing
upstream are implemented locally and marked `# TEMPORARY: upstream issue`
per docs/001-IMPLEMENTATION-PLAN.md §3.1.

Intended interface (§3.2):
    g = WireGraph.build(features, k, seed)
    L  = g.laplacian(normalized=True)      # torch.sparse_coo_tensor, cached
    R  = g.rayleigh_quotient(f)            # λτ / Dirichlet-energy proxy
    z  = g.spectral_embedding(n_components=2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

try:
    import arrowspace as asp  # type: ignore
    _HAS_ASP = True
except Exception:
    asp = None  # type: ignore
    _HAS_ASP = False


def _normalize_symmetric(L: Tensor) -> Tensor:
    """L_sym = D^{-1/2} L D^{-1/2} (symmetric normalised Laplacian)."""
    d = L.diagonal().clamp(min=0.0)
    d_inv_sqrt = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    D = torch.diag(d_inv_sqrt)
    return D @ L @ D


@dataclass
class WireGraph:
    """Feature-space graph Laplacian built via ArrowSpace.

    The Laplacian is (F, F) where F is the feature dimension of the input
    data. ArrowSpace builds the graph from item co-occurrence structure:
    pass data as (N, F) where N = items, F = features.
    """
    laplacian_dense: Tensor          # (F, F)
    laplacian_sym: Tensor            # (F, F) normalised, cached
    features: Tensor                 # (N, D) source items
    nnz: int

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, features: Tensor, k: int = 4, seed: int = 3407, mode: str = "feature") -> "WireGraph":
        """Build a graph Laplacian via ArrowSpace.

        Parameters
        ----------
        features : Tensor (N, F)
            Item-feature matrix. N items, F features.
        k : int
            Max clusters for ArrowSpace builder (feature mode) or k-NN
            neighbors (item mode). Default 4.
        seed : int
            Random seed.
        mode : str
            "feature" — ArrowSpace returns (F, F) feature-space Laplacian.
            "item" — k-NN Laplacian on item rows, returns (N, N).
        """
        if mode == "feature" and _HAS_ASP:
            items_np = features.detach().cpu().to(torch.float64).numpy()
            L = cls._arrowspace_laplacian(items_np, k=k, seed=seed)
        else:
            L = cls._knn_laplacian_fallback(features, k=k, seed=seed)
        L = L.to(dtype=torch.float32)
        L_sym = _normalize_symmetric(L)
        return cls(laplacian_dense=L, laplacian_sym=L_sym, features=features, nnz=int((L != 0).sum()))

    @staticmethod
    def _arrowspace_laplacian(items_np: np.ndarray, k: int, seed: int) -> Tensor:
        """Build L_F via the real ArrowSpace Rust bindings."""
        builder = (asp.ArrowSpaceBuilder()
                   .with_seed(seed)
                   .with_cluster_max_clusters(k))
        _space, gl = builder.build_full({}, items_np)
        L_np = np.array(gl.to_dense())
        return torch.from_numpy(L_np)

    @staticmethod
    def _knn_laplacian_fallback(features: Tensor, k: int, seed: int) -> Tensor:
        """Fallback k-NN Laplacian when ArrowSpace bindings are absent."""
        n = features.shape[0]
        f = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        sims = f @ f.t()
        sims.fill_diagonal_(-np.inf)
        k = min(k, n - 1)
        vals, idx = sims.topk(k, dim=-1)
        weights = torch.clamp(vals, min=0.0)
        A = torch.zeros(n, n)
        A.scatter_(1, idx, weights)
        A = (A + A.t()) * 0.5
        D = A.sum(dim=-1)
        L = torch.diag(D) - A
        return L

    # ------------------------------------------------------------------ #
    def laplacian(self, normalized: bool = True, sparse: bool = True) -> Tensor:
        L = self.laplacian_sym if normalized else self.laplacian_dense
        if not sparse:
            return L
        i = L.nonzero(as_tuple=False).t()
        v = L[i[0], i[1]]
        return torch.sparse_coo_tensor(i, v, L.shape).coalesce()

    def rayleigh_quotient(self, f: Tensor) -> Tensor:
        # R(f) = f^T L f / f^T f  — the λτ / Dirichlet-energy proxy.
        # TEMPORARY: upstream issue — arrowspace has no `taumode_score`.
        f = f.to(dtype=self.laplacian_dense.dtype, device=self.laplacian_dense.device)
        Lf = self.laplacian_dense @ f
        den = (f * f).sum(dim=-1) + 1e-8
        # per-node Rayleigh energy density R_k(f) = f_k (L f)_k / (f^T f)
        return f * Lf / den

    def spectral_embedding(self, n_components: int = 2) -> Tensor:
        vals, vecs = torch.linalg.eigh(self.laplacian_sym)
        # skip the trivial zero eigenvalue (smallest), take the next ones
        start = 1
        z = vecs[:, start:start + n_components]
        return z
