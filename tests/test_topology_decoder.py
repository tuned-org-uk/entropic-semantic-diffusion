"""
Tests for topology-adaptive spectral decoder (issue #24).
"""
from __future__ import annotations

import torch

from esdm.models.topology_decoder import (
    TopologyDecoder,
    heat_kernel_sample,
    topology_reverse_sde,
)
from esdm.models.esdm import ESDM


def _build_small_model():
    F = 16
    cfg = {
        "model": {
            "dim_input": F, "dim_latent": F // 4, "n_nodes": F,
            "n_modes": F // 4, "n_vibrational_steps": 2, "gamma": 0.1,
            "lam_s": 1.0, "sigma_sq": 1.0, "freeze_h": False,
        },
        "clock": {"eps": 1e-8, "heat_death_patience": 200, "eps_rel": 0.0},
        "training": {"batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "max_steps": 10},
    }
    A = torch.rand(F, F)
    A = (A + A.T) / 2
    A.fill_diagonal_(0)
    D = A.sum(dim=-1).diag()
    L = D - A
    cfg["model"]["L_base"] = L
    model = ESDM(cfg)
    x = torch.randn(4, F)
    with torch.no_grad():
        _ = model(x)
    return model


def test_topology_decoder_shape():
    model = _build_small_model()
    phi = model.U_q  # (F, q)
    dec = TopologyDecoder(phi)
    Q = torch.randn(4, 16)
    x_hat = dec(Q)
    assert x_hat.shape == (4, 16)
    assert torch.all(torch.isfinite(x_hat))


def test_topology_decoder_spectral_coords():
    model = _build_small_model()
    phi = model.U_q
    dec = TopologyDecoder(phi[:, :4])
    Q = torch.randn(4, 16)
    q_hat = dec.encode_to_spectral(Q)
    assert q_hat.shape == (4, 4)


def test_heat_kernel_sample_shape():
    K = 8
    L_K = torch.randn(K, K)
    L_K = L_K @ L_K.T  # PSD
    samples = heat_kernel_sample(L_K, t=1.0, n_samples=16, seed=3407)
    assert samples.shape == (16, K)
    assert torch.all(torch.isfinite(samples))


def test_heat_kernel_sample_reproducible():
    K = 8
    L_K = torch.randn(K, K)
    L_K = L_K @ L_K.T
    s1 = heat_kernel_sample(L_K, t=1.0, n_samples=8, seed=3407)
    s2 = heat_kernel_sample(L_K, t=1.0, n_samples=8, seed=3407)
    assert torch.allclose(s1, s2, atol=1e-4)


def test_topology_reverse_sde_shape():
    model = _build_small_model()
    samples = topology_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert samples.shape == (4, 16)
    assert torch.all(torch.isfinite(samples))


def test_topology_reverse_sde_normalized():
    model = _build_small_model()
    samples = topology_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    norms = samples.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_topology_reverse_sde_reproducible():
    model = _build_small_model()
    s1 = topology_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    s2 = topology_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert torch.allclose(s1, s2, atol=1e-4)


def test_topology_reverse_sde_restores_state():
    model = _build_small_model()
    tau_before = float(model.clock.tau_n.item())
    qtm1_before = model.vibrational.Q_tm1.clone()
    _ = topology_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert float(model.clock.tau_n.item()) == tau_before
    assert torch.equal(model.vibrational.Q_tm1, qtm1_before)


# ---------------------------------------------------------------------------
# Projected Laplacian and empirical prior tests
# ---------------------------------------------------------------------------


def test_build_projected_laplacian_shape():
    """L_K = U_q^T L_F U_q has shape (K, K)."""
    from esdm.models.topology_decoder import build_projected_laplacian
    F, K = 16, 4
    U_q = torch.randn(F, K)
    L_F = torch.randn(F, F)
    L_F = L_F @ L_F.T  # PSD
    L_K = build_projected_laplacian(U_q, L_F)
    assert L_K.shape == (K, K)
    assert torch.allclose(L_K, L_K.T, atol=1e-5)  # symmetric


def test_build_projected_laplacian_psd():
    """L_K is positive semi-definite when L_F is PSD."""
    from esdm.models.topology_decoder import build_projected_laplacian
    F, K = 8, 4
    torch.manual_seed(3407)
    A = torch.rand(F, F)
    A = (A + A.T) / 2
    A.fill_diagonal_(0)
    L_F = torch.diag(A.sum(dim=-1)) - A  # graph Laplacian (PSD by construction)
    _, vecs = torch.linalg.eigh(L_F.double())
    U_q = vecs.float()[:, :K]
    L_K = build_projected_laplacian(U_q, L_F)
    evals = torch.linalg.eigvalsh(L_K)
    assert evals.min() >= -1e-5  # PSD


def test_empirical_spectral_prior_shape():
    """Empirical prior returns mean (K,) and covariance (K, K)."""
    from esdm.models.topology_decoder import empirical_spectral_prior
    F, K, N = 16, 4, 50
    torch.manual_seed(3407)
    U_q = torch.randn(F, K)
    data = torch.nn.functional.normalize(torch.randn(N, F), dim=-1)
    mean, cov = empirical_spectral_prior(U_q, data)
    assert mean.shape == (K,)
    assert cov.shape == (K, K)


def test_empirical_spectral_prior_with_lambdas():
    """Lambda weighting biases the prior toward low-lambda (smooth) items."""
    from esdm.models.topology_decoder import empirical_spectral_prior
    F, K, N = 16, 4, 50
    torch.manual_seed(3407)
    U_q = torch.randn(F, K)
    data = torch.nn.functional.normalize(torch.randn(N, F), dim=-1)
    lambdas = torch.rand(N)  # uniform [0, 1]

    # Weighted prior (low lambda = high weight)
    mean_w, cov_w = empirical_spectral_prior(U_q, data, lambdas=lambdas)
    # Unweighted prior
    mean_u, cov_u = empirical_spectral_prior(U_q, data, lambdas=None)

    # The weighted and unweighted means should differ (different weighting).
    assert not torch.allclose(mean_w, mean_u, atol=1e-6)
    # Both covariances should be PSD.
    assert torch.linalg.eigvalsh(cov_w).min() >= -1e-5
    assert torch.linalg.eigvalsh(cov_u).min() >= -1e-5


def test_sample_from_prior_shape():
    """Sampling from the empirical prior produces (n_samples, K)."""
    from esdm.models.topology_decoder import sample_from_prior
    K = 4
    mean = torch.randn(K)
    cov = torch.randn(K, K)
    cov = cov @ cov.T + 1e-3 * torch.eye(K)
    samples = sample_from_prior(mean, cov, n_samples=8, seed=3407)
    assert samples.shape == (8, K)
    assert torch.all(torch.isfinite(samples))
