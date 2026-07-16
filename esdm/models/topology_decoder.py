"""
Topology-adaptive decoder for ArrowSpace generation (issue #24).

ArrowSpace is the space defined by:
- The graph Laplacian L_F (the `gl` property) — structural connectivity
- The lambda distribution (the `lambdas()` method) — per-item energy scores

This module decodes from spectral coordinates on that space, not from the
ambient R^F. The spectral chart is defined by U_q (smooth eigenvectors of
L_F), and the prior on the chart combines the graph geometry (via the
projected Laplacian L_K = U_q^T L_F U_q) with the data's energy
distribution (via the lambdas).

Sampling in spectral coordinates is sampling on the manifold by
construction — a point drawn from the prior and lifted back via U_q
always lies on the smooth submanifold. Off-manifold generation is
geometrically impossible.

See docs/00-DESIGN.md S12 (2.5-D Dual-Space Track).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TopologyDecoder(nn.Module):
    """
    Decodes from K-dim spectral coordinates to F-dim embeddings.

    Projects Q onto the K smooth eigenvectors of L_F (U_q), then maps
    the spectral coefficients through an MLP. The smooth eigenvectors
    define the ArrowSpace spectral chart — the low-dimensional manifold
    on which the data lives.

    Parameters
    ----------
    phi : Tensor (F, K)
        Leading K eigenvectors of L_F (smooth modes, lowest eigenvalues).
        Typically model.U_q.
    out_dim : int
        Output embedding dimension (defaults to F).
    hidden_dim : int
        MLP hidden width.
    """

    def __init__(
        self,
        phi: Tensor,
        out_dim: int | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        F_dim, K = phi.shape
        self.F_dim = F_dim
        self.K = K
        self.out_dim = out_dim if out_dim is not None else F_dim

        self.register_buffer("phi", phi)  # (F, K)

        self.mlp = nn.Sequential(
            nn.Linear(self.K, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def encode_to_spectral(self, Q: Tensor) -> Tensor:
        """Project Q onto the smooth eigenbasis. Q: (B, F) -> q_hat: (B, K)."""
        return Q @ self.phi  # (B, K)

    def lift_from_spectral(self, q_hat: Tensor) -> Tensor:
        """Lift spectral coefficients to ambient space. q_hat: (B, K) -> Q: (B, F)."""
        return q_hat @ self.phi.T  # (B, F)

    def forward(self, Q: Tensor) -> Tensor:
        """Q: (B, F) -> x_hat: (B, out_dim). Projects to spectral coords, decodes via MLP."""
        q_hat = self.encode_to_spectral(Q)  # (B, K)
        return self.mlp(q_hat)  # (B, out_dim)


def build_projected_laplacian(U_q: Tensor, L_F: Tensor) -> Tensor:
    """
    Projected (reduced) Laplacian in the spectral chart.

    L_K = U_q^T L_F U_q  in R^{K x K}

    This is the ArrowSpace graph structure projected onto the smooth
    eigenbasis. Its eigenvalues encode the energy landscape of the
    spectral chart — low eigenvalues are smooth directions, high
    eigenvalues are rough directions.
    """
    return U_q.T @ L_F @ U_q  # (K, K)


def heat_kernel_sample(
    L_K: Tensor,
    t: Tensor | float,
    n_samples: int = 64,
    seed: int = 3407,
    lambdas: Tensor | None = None,
) -> Tensor:
    """
    Draw samples from the heat kernel prior on the spectral chart.

    The prior is N(0, exp(-2t * L_K)) where L_K is the projected Laplacian.
    At t->0, samples concentrate near the origin (data-like). At t->inf,
    samples spread over the full smooth submanifold.

    When ``lambdas`` is provided (per-mode ArrowSpace energy scores in [0,1],
    projected onto the K-dim spectral chart), the prior covariance is
    weighted: low-lambda (smooth) modes get higher variance, high-lambda
    (rough) modes get suppressed. This biases sampling toward the regions
    where real data concentrates on the energy landscape.

    Parameters
    ----------
    L_K : Tensor (K, K)
        Projected Laplacian in the spectral chart.
    t : float or Tensor
        Diffusion time.
    n_samples : int
    seed : int
    lambdas : Tensor (K,), optional
        Per-mode ArrowSpace lambda scores in [0,1]. Low = smooth (on-manifold).

    Returns
    -------
    Tensor (n_samples, K) — spectral coordinates on the manifold.
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(float(t), dtype=L_K.dtype, device=L_K.device)

    gen = torch.Generator(device=L_K.device).manual_seed(seed)
    K = L_K.shape[0]

    cov = torch.matrix_exp(-2.0 * t * L_K)

    if lambdas is not None:
        lambdas = lambdas.to(dtype=cov.dtype, device=cov.device)
        weight = torch.exp(-lambdas * 5.0)  # exponential decay: low lambda = high weight
        weight = weight / weight.sum() * K  # normalise so mean weight = 1
        cov = cov * weight.unsqueeze(0) * weight.unsqueeze(1)

    cov = cov + 1e-6 * torch.eye(K, dtype=cov.dtype, device=cov.device)

    try:
        L_chol = torch.linalg.cholesky(cov)
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp(min=0)
        L_chol = evecs * evals.sqrt().unsqueeze(0)

    z = torch.randn(n_samples, K, dtype=L_K.dtype, device=L_K.device, generator=gen)
    return z @ L_chol.T  # (n_samples, K)


def empirical_spectral_prior(
    U_q: Tensor,
    data: Tensor,
    lambdas: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Empirical prior from training data projected onto the spectral chart.

    Computes the mean and covariance of the data's spectral coordinates.
    If lambdas are provided, uses them as per-item weights (low lambda =
    smooth = higher weight) to bias the prior toward the smooth regime.

    Parameters
    ----------
    U_q : Tensor (F, K)
        Smooth eigenvectors of L_F.
    data : Tensor (N, F)
        Training embeddings (L2-normalised).
    lambdas : Tensor (N,), optional
        Per-item ArrowSpace lambda scores in [0, 1]. Low = smooth.

    Returns
    -------
    (mean, cov) : (Tensor (K,), Tensor (K, K))
        Weighted mean and covariance of spectral coordinates.
    """
    q_hat = data @ U_q  # (N, K)

    if lambdas is not None:
        lambdas = lambdas.to(dtype=q_hat.dtype, device=q_hat.device)
        weights = (1.0 - lambdas).clamp(min=1e-8)  # low lambda = high weight
        weights = weights / weights.sum()
        mean = (q_hat * weights.unsqueeze(-1)).sum(dim=0)  # (K,)
        centered = q_hat - mean.unsqueeze(0)
        cov = (centered * weights.unsqueeze(-1)).T @ centered  # (K, K)
    else:
        mean = q_hat.mean(dim=0)  # (K,)
        centered = q_hat - mean.unsqueeze(0)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)

    cov = cov + 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return mean, cov


def sample_from_prior(
    mean: Tensor,
    cov: Tensor,
    n_samples: int = 64,
    seed: int = 3407,
) -> Tensor:
    """Draw samples from a Gaussian prior on the spectral chart."""
    gen = torch.Generator(device=cov.device).manual_seed(seed)
    K = cov.shape[0]

    try:
        L_chol = torch.linalg.cholesky(cov)
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp(min=0)
        L_chol = evecs * evals.sqrt().unsqueeze(0)

    z = torch.randn(n_samples, K, dtype=cov.dtype, device=cov.device, generator=gen)
    return mean.unsqueeze(0) + z @ L_chol.T  # (n_samples, K)


def topology_reverse_sde(
    model,
    n_samples: int = 64,
    n_steps: int = 100,
    seed: int = 3407,
    device: torch.device | None = None,
) -> Tensor:
    """
    Sample from a trained ESDM via ArrowSpace topology-aware reverse diffusion.

    Instead of operating in ambient R^F, this sampler:
    1. Builds the spectral chart from the model's U_q (smooth eigenvectors)
       and L_F (graph Laplacian) — the ArrowSpace structure
    2. Draws initial spectral coordinates from the heat kernel prior
       N(0, exp(-2t * L_K)) where L_K = U_q^T L_F U_q
    3. At each reverse step: lift to ambient space, run model forward pass
       as score predictor, project back to spectral coordinates
    4. Final lift gives the generated embedding on the smooth submanifold

    Points drawn from the spectral chart always lie on the smooth
    submanifold — off-manifold generation is geometrically impossible.
    """
    if device is None:
        device = next(model.parameters()).device

    gen = torch.Generator(device=device).manual_seed(seed)
    model.eval()

    # Save model state.
    clock = model.clock
    saved_tau_n = float(clock.tau_n.item())
    saved_tau_max = float(clock.tau_max_est.item())
    saved_prev_kl = float(clock._prev_kl_tau.item())
    saved_flat = int(clock._flat_steps.item())
    saved_prev_phi = float(clock._prev_phi.item())
    saved_rho_plus = model._last_rho_plus.clone()
    saved_rho_minus = model._last_rho_minus.clone()
    saved_N_active = int(model._last_N_active.item())
    saved_Q_tm1 = model.vibrational.Q_tm1.clone()

    tau_max = saved_tau_max if saved_tau_max > 0 else 1.0
    taus = torch.linspace(tau_max, 1e-6, n_steps, device=device)

    # Build the ArrowSpace spectral chart.
    U_q = model.U_q  # (F, K) — smooth eigenvectors
    L_F = model.L_base  # (F, F) — graph Laplacian
    L_K = build_projected_laplacian(U_q, L_F)  # (K, K)

    # Project ArrowSpace lambdas onto the K-dim spectral chart.
    # model.lambdas holds per-feature (F,) energy scores from
    # aspace.lambdas(). We project to per-mode (K,) by energy-weighting
    # each eigenvector: lambda_K^(k) = sum_f lambda_f * U_{f,k}^2.
    # Low lambda_K = smooth mode = high variance in the prior.
    lambdas_K = None
    if getattr(model, "lambdas", None) is not None:
        lambdas_F = model.lambdas.to(U_q.dtype)
        lambdas_K = (lambdas_F.unsqueeze(1) * U_q.pow(2)).sum(dim=0)  # (K,)
        lmax = lambdas_K.max().clamp(min=1e-8)
        lambdas_K = lambdas_K / lmax  # normalise to [0, 1]

    # Initialise from the heat kernel prior at t = tau_max.
    q_hat = heat_kernel_sample(L_K, float(tau_max), n_samples, seed, lambdas=lambdas_K)
    q_hat = q_hat.to(device=device, dtype=U_q.dtype)

    for i in range(n_steps - 1):
        tau_cur = float(taus[i])
        tau_next = float(taus[i + 1])

        clock.tau_n.fill_(tau_cur)
        clock.tau_max_est.fill_(tau_max)

        ab_next = max(float(clock.alpha_bar(tau=torch.tensor(tau_next, device=device)).item()), 1e-8)

        # Lift spectral coords to ambient space and normalise.
        Q = q_hat @ U_q.T  # (n_samples, F)
        Q = F.normalize(Q, dim=-1)

        # Run model forward pass as score predictor.
        torch.manual_seed(seed + i)
        with torch.no_grad():
            _ = model(Q)
            Q_predicted = model._last_Q  # (B, F)
        Q_predicted = F.normalize(Q_predicted, dim=-1)

        # Project predicted Q back to spectral coordinates.
        q_pred = Q_predicted @ U_q  # (n_samples, K)

        # DDIM-style update in spectral coordinates.
        # Noise is scaled by lambda weights: more noise in smooth (low-lambda)
        # directions, less in rough (high-lambda) directions.
        z_noise = torch.randn(n_samples, U_q.shape[1], device=device, generator=gen)
        if lambdas_K is not None:
            noise_weight = torch.exp(-lambdas_K.to(device) * 5.0)
            noise_weight = noise_weight / noise_weight.sum() * len(noise_weight)
            z_noise = z_noise * noise_weight.unsqueeze(0)
        a_sqrt_next = ab_next ** 0.5
        noise_sqrt_next = (1.0 - ab_next) ** 0.5
        q_hat = a_sqrt_next * q_pred + noise_sqrt_next * z_noise

    # Final lift: spectral coords -> ambient space -> L2-normalised embedding.
    Q_final = q_hat @ U_q.T
    result = F.normalize(Q_final, dim=-1)

    # Restore model state.
    clock.tau_n.fill_(saved_tau_n)
    clock.tau_max_est.fill_(saved_tau_max)
    clock._prev_kl_tau.fill_(saved_prev_kl)
    clock._flat_steps.fill_(saved_flat)
    clock._prev_phi.fill_(saved_prev_phi)
    model._last_rho_plus = saved_rho_plus
    model._last_rho_minus = saved_rho_minus
    model._last_N_active.fill_(saved_N_active)
    model.vibrational.Q_tm1 = saved_Q_tm1

    return result
