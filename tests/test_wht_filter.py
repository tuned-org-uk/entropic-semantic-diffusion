"""
Tests for the vendored Graph-WHT spectral filter and its integration into
``DualSpaceMatrix.build()``.
"""
import numpy as np
import pytest
import torch

from esdm.laplacian import DualSpaceMatrix
from esdm.wht_filter import GraphWHTFilter, heat_kernel


@pytest.fixture()
def L_F():
    """Small symmetric PSD feature-space Laplacian."""
    F = 16
    rng = np.random.default_rng(3407)
    A = rng.standard_normal((F, F))
    L = A.T @ A
    # Make it Laplacian-like: zero row/col sums on the diagonal part.
    np.fill_diagonal(L, L.diagonal() + 1e-3)
    return (L + L.T) / 2.0


@pytest.fixture()
def V_and_LF():
    """Corpus embeddings and feature-space Laplacian for DualSpaceMatrix."""
    torch.manual_seed(3407)
    np.random.seed(3407)
    N, F = 50, 32
    V = torch.randn(N, F)
    A = torch.randn(F, F)
    L_F = A.T @ A
    return V, L_F


def test_graph_wht_filter_apply_shape(L_F):
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    x = np.random.default_rng(3407).standard_normal(L_F.shape[0])
    x_smooth = filt.apply(x, heat_kernel(t=1.0))
    assert x_smooth.shape == x.shape


def test_graph_wht_filter_preserves_constant_signal(L_F):
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    x = np.ones(L_F.shape[0])
    # Heat kernel at λ=0 is 1, so the DC component is preserved.
    x_smooth = filt.apply(x, heat_kernel(t=1.0))
    assert np.allclose(x_smooth, x, atol=1e-10)


def test_filter_matrix_symmetry(L_F):
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    L_smooth = filt.filter_matrix(L_F, t=1.0)
    assert L_smooth.shape == L_F.shape
    assert np.allclose(L_smooth, L_smooth.T, atol=1e-12)


def test_dual_space_matrix_build_with_wht_filter(V_and_LF):
    V, L_F = V_and_LF
    filt = GraphWHTFilter(L_matrix=L_F.detach().cpu().numpy()).fit()

    dsm = DualSpaceMatrix(alpha=0.5, beta=0.5)
    dsm.build(V, L_F, wht_filter=filt, wht_t=1.0)

    assert dsm.M is not None
    assert torch.allclose(dsm.M, dsm.M.T, atol=1e-6)
    assert torch.all(torch.isfinite(dsm.eigvals))


def test_dual_space_matrix_wht_filter_changes_semantic_term(V_and_LF):
    V, L_F = V_and_LF
    filt = GraphWHTFilter(L_matrix=L_F.detach().cpu().numpy()).fit()

    dsm_raw = DualSpaceMatrix(alpha=0.0, beta=1.0)
    dsm_raw.build(V, L_F)

    dsm_smooth = DualSpaceMatrix(alpha=0.0, beta=1.0)
    dsm_smooth.build(V, L_F, wht_filter=filt, wht_t=1.0)

    # The smoothed semantic term should differ from the raw one.
    assert not torch.allclose(dsm_raw.M, dsm_smooth.M, atol=1e-6)
