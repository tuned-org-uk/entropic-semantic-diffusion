"""
esdm/wht_filter.py
==================
Vendored copy of the Graph Walsh-Hadamard Transform (Graph-WHT) spectral
filter from ``pyarrowspace`` PR #21 (arxiv:2511.07347).

This module is a stop-gap: once PR #21 lands and ``GraphWHTFilter`` becomes
a proper export of ``pyarrowspace``, this file should be removed and
``esdm.laplacian`` should import directly from upstream.

Python-layer Graph Walsh-Hadamard Transform (Graph-WHT) spectral filter.
This module implements the Graph-WHT step for the Spectral Diffusion pipeline,
operating on the feature-space Laplacian L_F stored by ArrowSpace.

No Rust changes required: all computation is pure NumPy / SciPy on top of
the existing ArrowSpaceBuilder / GraphLaplacian Python interface.

Concrete definition
-------------------
Given:
  L_F  - feature-space graph Laplacian (F×F, extracted from ArrowSpace storage)
  x    - signal vector of length F  (query embedding or diffusion latent state)
  h(λ) - spectral filter function   (default: low-pass heat kernel  h=exp(-t·λ))

Step 1  Spectral decomposition of L_F
    L_F · u_k = λ_k · u_k,   k = 0 … F-1
    U  = [u_0 | … | u_{F-1}]          (F×F eigenvector matrix)

Step 2  Fiedler ordering
    Sort nodes by u_1 (Fiedler vector) so spatial adjacency aligns with
    spectral frequency before the Hadamard butterfly.

Step 3  WHT in Fiedler order
    Pad x_ordered to the next power of 2 → length P
    H_P = (1/sqrt(P)) · recursive Hadamard (sequency/Walsh ordering)
    X_wht = H_P · x_padded

Step 4  Spectral filter in WHT domain
    Approximate WHT eigenvalues (sequency):  λ_k_approx = (2/P)·k
    Filter:   X_filtered_k = h(λ_k_approx) · X_wht_k

Step 5  Inverse WHT + unpad + undo Fiedler order
    x_out = IWHT(X_filtered)[:F]  →  x_reconstructed[inverse_fiedler_order]
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: fast in-place sequency-ordered Walsh-Hadamard Transform
# ---------------------------------------------------------------------------


def _wht_sequency_inplace(x: np.ndarray) -> np.ndarray:
    """
    In-place sequency-ordered Walsh-Hadamard Transform.
    Input length must be a power of 2.
    Returns the transformed array (same object, normalised by 1/sqrt(N)).
    """
    N = len(x)
    assert N & (N - 1) == 0, "WHT length must be a power of 2"

    # Standard (natural-order) Fast WHT  ----------------------------------
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2

    # Bit-reversal permutation to reach sequency order  -------------------
    log2n = int(np.log2(N))
    perm = np.arange(N, dtype=np.int32)
    # Gray-code reorder: sequency index k is the bit-reversal of gray(k)
    gray = perm ^ (perm >> 1)
    bits_reversed = np.zeros(N, dtype=np.int32)
    for bit in range(log2n):
        bits_reversed |= ((gray >> bit) & 1) << (log2n - 1 - bit)
    x[:] = x[bits_reversed]

    x /= np.sqrt(N)
    return x


def _iwht_sequency_inplace(x: np.ndarray) -> np.ndarray:
    """Inverse sequency WHT.  WHT is self-inverse up to normalisation."""
    # Undo the sequency permutation first, then apply forward WHT again.
    # Since H = H^T and H·H = N·I  →  H^{-1} = (1/N)·H
    # Because we normalise by 1/sqrt(N) in forward,  IWHT = WHT as well.
    return _wht_sequency_inplace(x)

def _wht_matrix_inplace(X: np.ndarray) -> np.ndarray:
    """
    In-place sequency-ordered Walsh-Hadamard Transform along the last axis.
    Input shape (... , P) where P is a power of 2.
    Returns the transformed array (same object, normalised by 1/sqrt(P)).
    """
    P = X.shape[-1]
    assert P & (P - 1) == 0, "WHT length must be a power of 2"

    # Standard (natural-order) Fast WHT along last axis -------------------
    h = 1
    while h < P:
        step = h * 2
        for i in range(0, P, step):
            j_slice = slice(i, i + h)
            k_slice = slice(i + h, i + step)
            a = X[..., j_slice]
            b = X[..., k_slice]
            # Materialise sums/differences before writing back, because a and
            # b are views into X.
            a_plus_b = a + b
            a_minus_b = a - b
            X[..., j_slice] = a_plus_b
            X[..., k_slice] = a_minus_b
        h *= 2

    # Bit-reversal permutation to reach sequency order --------------------
    log2p = int(np.log2(P))
    perm = np.arange(P, dtype=np.int32)
    gray = perm ^ (perm >> 1)
    bits_reversed = np.zeros(P, dtype=np.int32)
    for bit in range(log2p):
        bits_reversed |= ((gray >> bit) & 1) << (log2p - 1 - bit)
    X[..., :] = X[..., bits_reversed]

    X /= np.sqrt(P)
    return X


def _iwht_matrix_inplace(X: np.ndarray) -> np.ndarray:
    """Inverse sequency WHT along the last axis."""
    return _wht_matrix_inplace(X)



# ---------------------------------------------------------------------------
# Fiedler ordering helpers
# ---------------------------------------------------------------------------


def _fiedler_order(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Fiedler ordering of nodes in L.

    The Fiedler vector (second smallest eigenvector of L) induces a 1-D
    embedding that respects graph connectivity.  Sorting nodes by this
    value groups adjacent nodes together, aligning the WHT butterfly with
    spectral frequency.

    Parameters
    ----------
    L : (F, F) dense or sparse symmetric Laplacian.

    Returns
    -------
    order         : argsort of Fiedler vector  (node → position)
    inverse_order : undo the permutation       (position → node)
    eigenvalues   : true Laplacian eigenvalues
    eigenvectors  : true Laplacian eigenvectors
    """
    if issparse(L):
        L = L.toarray()  # type: ignore[attr-defined]
    L_dense = np.asarray(L, dtype=np.float64)

    L_sym = (L_dense + L_dense.T) / 2.0          # enforce symmetry
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    # Smallest eigenvalue ≈ 0 (constant vector); Fiedler = index 1
    fiedler_vec = eigenvectors[:, 1]
    order = np.argsort(fiedler_vec)
    inverse_order = np.argsort(order)
    return order, inverse_order, eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Built-in filter kernels
# ---------------------------------------------------------------------------


def heat_kernel(t: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Low-pass heat / diffusion kernel:  h(λ) = exp(-t·λ)."""

    def _h(lambdas: np.ndarray) -> np.ndarray:
        return np.exp(-t * lambdas)

    return _h


def ideal_lowpass(cutoff: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    """Hard low-pass: h(λ) = 1 if λ ≤ cutoff, else 0."""

    def _h(lambdas: np.ndarray) -> np.ndarray:
        return (lambdas <= cutoff).astype(np.float64)

    return _h


def bandpass(
    low: float = 0.1, high: float = 0.8
) -> Callable[[np.ndarray], np.ndarray]:
    """Band-pass: h(λ) = 1 if low ≤ λ ≤ high, else 0."""

    def _h(lambdas: np.ndarray) -> np.ndarray:
        return ((lambdas >= low) & (lambdas <= high)).astype(np.float64)

    return _h


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class GraphWHTFilter:
    """
    Graph-WHT spectral filter operating on the feature-space Laplacian L_F
    stored by an ArrowSpace instance.

    Usage
    -----
    >>> from arrowspace import ArrowSpaceBuilder
    >>> from esdm.wht_filter import GraphWHTFilter, heat_kernel
    >>>
    >>> aspace, gl = ArrowSpaceBuilder().build_and_store(graph_params, items)
    >>> filt = GraphWHTFilter(gl)          # pass the GraphLaplacian object
    >>> filt.fit()                         # precompute Fiedler ordering
    >>>
    >>> x = items[0]                       # raw feature vector
    >>> x_smooth = filt.apply(x, heat_kernel(t=1.0))

    Attributes
    ----------
    F             : feature dimension
    P             : next power-of-2 padding length
    fiedler_order : node-to-position permutation
    inv_order     : position-to-node permutation
    eigenvalues   : true Laplacian eigenvalues (used for diagnostics)
    wht_lambdas   : approximate WHT sequency eigenvalues in [0, 2]
    """

    def __init__(
        self,
        graph_laplacian=None,
        L_matrix: Optional[np.ndarray] = None,
        a_tau_scale: float = 5.0,
        tau_decay_power: float = 1.0,
    ):
        """
        Parameters
        ----------
        graph_laplacian : GraphLaplacian, optional
            The object returned by ArrowSpaceBuilder.build_and_store().
            We call .matrix() (or fall back to .to_dense()) to get L_F.
        L_matrix : np.ndarray, optional
            Pass a prebuilt dense L_F directly (e.g. from the analyzer).
            If provided, graph_laplacian is ignored for the matrix.
        """
        self._gl = graph_laplacian
        self._L_matrix = L_matrix
        self.a_tau_scale = float(a_tau_scale)
        self.tau_decay_power = float(tau_decay_power)
        self.F: Optional[int] = None
        self.P: Optional[int] = None
        self.fiedler_order: Optional[np.ndarray] = None
        self.inv_order: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.wht_lambdas: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "GraphWHTFilter":
        """
        Precompute Fiedler ordering, WHT padding, and approximate eigenvalues.
        Must be called once before apply().
        """
        L = self._get_laplacian()
        self.F = L.shape[0]

        # Next power-of-2 for WHT
        self.P = 1
        while self.P < self.F:
            self.P <<= 1

        logger.info(
            "GraphWHTFilter.fit(): F=%d, P=%d (padding=%d)",
            self.F,
            self.P,
            self.P - self.F,
        )

        # Fiedler ordering + true eigenvalues (reused for diagnostics)
        (
            self.fiedler_order,
            self.inv_order,
            self.eigenvalues,
            self.eigenvectors,
        ) = _fiedler_order(L)

        # Approximate WHT sequency eigenvalues:  λ_k ≈ (2/P)·k  ∈ [0, 2]
        self.wht_lambdas = (2.0 / self.P) * np.arange(
            self.P, dtype=np.float64
        )

        self._fitted = True
        logger.info(
            "GraphWHTFilter fitted.  Spectral gap: %.6f",
            self.eigenvalues[1] - self.eigenvalues[0]
            if len(self.eigenvalues) > 1
            else 0.0,
        )
        return self

    def _compute_bivariate_gate(
        self,
        tau: float,
        tau_max: float,
        r_k: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute the Path-A bivariate gate ``g_k(tau, r_k)``.

        The gate is a soft switch over WHT sequency bins.  Early in entropic
        time the gate is open; as ``tau -> tau_max`` the clock-driven offset
        closes it regardless of residual magnitude, preserving the heat-death
        termination invariant.
        """
        assert self.P is not None
        assert self.fiedler_order is not None
        assert self.F is not None

        norm = float(tau) / max(float(tau_max), 1e-12)
        norm = float(np.clip(norm, 0.0, 1.0))

        decay = (1.0 - norm) ** self.tau_decay_power
        growth = norm ** self.tau_decay_power

        # a(tau) weights residual influence; b(tau) provides clock-driven closure.
        a = self.a_tau_scale * decay
        b = self.a_tau_scale * (decay - growth)

        # Map per-mode residuals into Fiedler-ordered / padded WHT positions.
        residual = np.zeros(self.P, dtype=np.float64)
        if r_k is not None:
            r = np.asarray(r_k, dtype=np.float64)
            K = min(len(r), self.F)
            residual[:K] = r[self.fiedler_order[:K]]

        # f(r) = log(1 + r); add b to close gates at heat death.
        log_residual = np.log1p(residual)
        gate = 1.0 / (1.0 + np.exp(-(a * log_residual + b)))
        return gate

    def apply(
        self,
        x: np.ndarray,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        t: float = 1.0,
        tau: Optional[float] = None,
        tau_max: Optional[float] = None,
        r_k: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the Graph-WHT spectral filter to signal vector x.

        Parameters
        ----------
        x : (F,) array   — signal in original node order
        h : callable     — filter function h(lambdas) → weights; defaults to
                           heat_kernel(t)
        t : float        — diffusion time (used only when h is None)

        Returns
        -------
        x_filtered : (F,) array in original node order
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .apply()")

        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) != self.F:
            raise ValueError(
                f"Expected signal of length {self.F}, got {len(x)}"
            )

        if h is None:
            h = heat_kernel(t)

        assert self.P is not None
        assert self.wht_lambdas is not None

        # Step 2: Fiedler re-ordering + zero-padding
        x_ordered = np.zeros(self.P, dtype=np.float64)
        x_ordered[: self.F] = x[self.fiedler_order]

        # Step 3: Forward WHT  (in-place, normalised)
        x_wht = _wht_sequency_inplace(x_ordered.copy())

        # Step 4: Spectral filter
        weights = h(self.wht_lambdas)               # (P,)
        if r_k is not None:
            if tau is None or tau_max is None:
                raise ValueError(
                    "tau and tau_max are required when r_k is provided"
                )
            gate = self._compute_bivariate_gate(tau, tau_max, r_k)
            weights = weights * gate
        x_filtered_wht = weights * x_wht

        # Step 5: Inverse WHT
        x_rec = _iwht_sequency_inplace(x_filtered_wht)

        # Unpad + undo Fiedler order
        x_rec_F = x_rec[: self.F]
        x_out = x_rec_F[self.inv_order]

        return x_out

    def filter_batch(
        self,
        X: np.ndarray,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        t: float = 1.0,
        tau: Optional[float] = None,
        tau_max: Optional[float] = None,
        r_k: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the filter to the last axis of array ``X``.

        ``X`` may be any shape whose last dimension equals ``F``.  The
        operation is vectorised over all leading axes, avoiding a Python
        loop over rows.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .filter_batch()")

        X = np.asarray(X, dtype=np.float64)
        if X.shape[-1] != self.F:
            raise ValueError(
                f"Expected last axis of size {self.F}, got {X.shape[-1]}"
            )

        if h is None:
            h = heat_kernel(t)

        assert self.P is not None
        assert self.wht_lambdas is not None
        assert self.fiedler_order is not None
        assert self.inv_order is not None

        # Fiedler re-ordering + zero-padding along the last axis.
        shape = X.shape[:-1] + (self.P,)
        X_ordered = np.zeros(shape, dtype=np.float64)
        X_ordered[..., : self.F] = X[..., self.fiedler_order]

        # Forward WHT on all leading axes at once.
        X_wht = _wht_matrix_inplace(X_ordered.copy())

        # Spectral filter along the last axis.
        weights = h(self.wht_lambdas)               # (P,)
        if r_k is not None:
            if tau is None or tau_max is None:
                raise ValueError(
                    "tau and tau_max are required when r_k is provided"
                )
            gate = self._compute_bivariate_gate(tau, tau_max, r_k)
            weights = weights * gate
        X_filtered_wht = weights * X_wht

        # Inverse WHT.
        X_rec = _iwht_matrix_inplace(X_filtered_wht)

        # Unpad + undo Fiedler order.
        X_rec_F = X_rec[..., : self.F]
        X_out = X_rec_F[..., self.inv_order]

        return X_out

    def forward(
        self,
        x: np.ndarray,
        tau: Optional[float] = None,
        tau_max: Optional[float] = None,
        r_k: Optional[np.ndarray] = None,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        t: float = 1.0,
    ) -> np.ndarray:
        """
        Convenience alias for the WHT filter with optional bivariate gate.

        ``x`` may be 1-D ``(F,)`` or N-D ``(..., F)``.  When ``r_k`` is not
        provided the gate is bypassed and the behaviour reduces to the
        original heat-kernel filter.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return self.apply(
                x, h=h, t=t, tau=tau, tau_max=tau_max, r_k=r_k
            )
        return self.filter_batch(
            x, h=h, t=t, tau=tau, tau_max=tau_max, r_k=r_k
        )

    def filter_matrix(
        self,
        L: np.ndarray,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        t: float = 1.0,
        symmetrise: bool = True,
    ) -> np.ndarray:
        """
        Apply the Graph-WHT filter to a feature-space matrix.

        Each row of ``L`` is treated as a signal over feature nodes and
        filtered; the operation is then repeated on the columns so both the
        row and column spaces are smoothed consistently.  Finally the result
        is symmetrised (``(A + A.T) / 2``) if ``symmetrise=True``.

        Parameters
        ----------
        L : (F, F) array
            Feature-space matrix (typically the feature-space Laplacian L_F).
        h : callable, optional
            Filter function; defaults to ``heat_kernel(t)``.
        t : float
            Diffusion time when ``h`` is not provided.
        symmetrise : bool
            If True (default), return ``(L_smooth + L_smooth.T) / 2`` so the
            filtered Laplacian remains exactly symmetric.

        Returns
        -------
        L_smooth : (F, F) array
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .filter_matrix()")

        L = np.asarray(L, dtype=np.float64)
        if L.shape != (self.F, self.F):
            raise ValueError(
                f"Expected matrix shape ({self.F}, {self.F}), got {L.shape}"
            )

        # Filter rows, then columns, to treat both index spaces consistently.
        row_smooth = self.filter_batch(L, h=h, t=t)
        col_smooth = self.filter_batch(row_smooth.T, h=h, t=t).T

        if symmetrise:
            return (col_smooth + col_smooth.T) / 2.0
        return col_smooth

    def spectral_energy(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral energy distribution of signal x over WHT frequencies.

        Returns
        -------
        wht_lambdas : (P,) approximate eigenvalues
        energy      : (P,) squared WHT coefficients
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before spectral_energy()")

        assert self.P is not None
        assert self.wht_lambdas is not None

        x = np.asarray(x, dtype=np.float64).ravel()
        x_ordered = np.zeros(self.P, dtype=np.float64)
        x_ordered[: self.F] = x[self.fiedler_order]
        x_wht = _wht_sequency_inplace(x_ordered.copy())
        return self.wht_lambdas, x_wht ** 2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_laplacian(self) -> np.ndarray:
        """Resolve L_F from stored matrix or GraphLaplacian object."""
        if self._L_matrix is not None:
            return self._L_matrix

        gl = self._gl
        # Try the most common ArrowSpace GraphLaplacian access patterns
        for attr in ("matrix", "to_dense", "dense"):
            if hasattr(gl, attr):
                candidate = getattr(gl, attr)
                L = candidate() if callable(candidate) else candidate
                if issparse(L):
                    return L.toarray().astype(np.float64)
                return np.asarray(L, dtype=np.float64)

        # Fallback: try __array__
        return np.asarray(gl, dtype=np.float64)
