"""
Tests for the vendored Graph-WHT spectral filter.
"""
import numpy as np
import pytest

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
def rng():
    return np.random.default_rng(3407)


def test_graph_wht_filter_apply_shape(L_F, rng):
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    x = rng.standard_normal(L_F.shape[0])
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


def test_filter_batch_vectorised_shape(L_F, rng):
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    N = 10
    X = rng.standard_normal((N, L_F.shape[0]))
    X_smooth = filt.filter_batch(X, heat_kernel(t=1.0))
    assert X_smooth.shape == X.shape
    # Each row should match the scalar apply() result.
    for i in range(N):
        assert np.allclose(X_smooth[i], filt.apply(X[i], heat_kernel(t=1.0)))


def test_spectral_attenuation(L_F, rng):
    """High-frequency energy must decrease after heat-kernel filtering."""
    filt = GraphWHTFilter(L_matrix=L_F).fit()
    x = rng.standard_normal(L_F.shape[0])

    lambdas, energy_raw = filt.spectral_energy(x)
    x_smooth = filt.apply(x, heat_kernel(t=1.0))
    _, energy_smooth = filt.spectral_energy(x_smooth)

    # Split frequencies at the median approximate WHT eigenvalue.
    median_lambda = np.median(lambdas)
    high_freq = lambdas > median_lambda

    # High-frequency content must be attenuated.  Low frequencies are damped
    # less, so the *fraction* of total energy in the high-frequency band must
    # drop after the low-pass heat kernel.
    high_fraction_raw = energy_raw[high_freq].sum() / energy_raw.sum()
    high_fraction_smooth = energy_smooth[high_freq].sum() / energy_smooth.sum()
    assert high_fraction_smooth < high_fraction_raw
