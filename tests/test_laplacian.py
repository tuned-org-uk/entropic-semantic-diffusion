"""
Phase 1 test — DualSpaceMatrix.
All assertions from 01-IMPLEMENTATION-PLAN.md §2.4.
"""
import pytest
import torch


@pytest.fixture()
def V_and_LF():
    torch.manual_seed(0)
    N, F = 50, 32
    V = torch.randn(N, F)
    # Symmetric PSD L_F: L = A^T A
    A = torch.randn(F, F)
    L_F = A.T @ A
    return V, L_F


def test_M_is_symmetric(V_and_LF):
    from esdm.graphs.laplacian import DualSpaceMatrix
    V, L_F = V_and_LF
    dsm = DualSpaceMatrix(alpha=0.5, beta=0.5)
    dsm.build(V, L_F)
    assert torch.allclose(dsm.M, dsm.M.T, atol=1e-6)


def test_eigvals_finite(V_and_LF):
    from esdm.graphs.laplacian import DualSpaceMatrix
    V, L_F = V_and_LF
    dsm = DualSpaceMatrix()
    dsm.build(V, L_F)
    assert dsm.eigvals is not None
    assert torch.all(torch.isfinite(dsm.eigvals))


def test_rayleigh_finite(V_and_LF):
    from esdm.graphs.laplacian import DualSpaceMatrix
    V, L_F = V_and_LF
    dsm = DualSpaceMatrix()
    dsm.build(V, L_F)
    x = torch.randn(50)
    x = x / x.norm()
    r = dsm.rayleigh(x)
    assert torch.isfinite(r)


def test_project_shape(V_and_LF):
    from esdm.graphs.laplacian import DualSpaceMatrix
    V, L_F = V_and_LF
    dsm = DualSpaceMatrix()
    dsm.build(V, L_F)
    x = torch.randn(10, 50)
    out = dsm.project(x, k=8)
    assert out.shape == (10, 8)
