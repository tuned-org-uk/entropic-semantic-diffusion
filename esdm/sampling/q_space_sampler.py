"""
Q-space reverse sampler for ESDM (issue #29 Phase 1, Step 5).

The DDPM sampler (entropic_sampler.py) operates in input space and
collapses to zero because ESDM's diffusion is internal to Q. This sampler
operates directly in Q-space using a score-approximation approach:

1. Initialise Q in the U_q spectral subspace (noise constrained to the
   geometric prior).
2. At each reverse step (tau_max -> 0, alpha_bar 0 -> 1):
   a. Decode Q to x_hat = sqrt(alpha_bar) * Q (diffusion decoder formula)
   b. L2-normalise x_hat (match training data distribution)
   c. Run model forward pass -> get Q_predicted (model._last_Q)
   d. DDIM-style update: Q_next = sqrt(ab_next) * Q_predicted
                            + sqrt(1 - ab_next) * noise
3. At tau=0 (alpha_bar=1): x_hat = Q_final (the generated embedding)

This does NOT algebraically invert the wave recurrence (both B_t and H are
nonlinear in Q_t). Instead it uses the trained model's forward pass as a
score predictor, evaluating at the current state (leapfrog approximation).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def q_space_reverse_sde(
    model,
    n_samples: int = 64,
    n_steps: int = 100,
    seed: int = 3407,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Sample from a trained ESDM via Q-space reverse diffusion.

    Parameters
    ----------
    model : ESDM
        Trained ESDM model. Must be in eval mode.
    n_samples : int
        Number of samples to generate.
    n_steps : int
        Number of reverse steps (more = higher quality, slower).
    seed : int
        Random seed for reproducibility.
    device : torch.device, optional

    Returns
    -------
    Tensor, shape (n_samples, dim_input)
        Generated embeddings in the data space (L2-normalised).
    """
    if device is None:
        device = next(model.parameters()).device

    gen = torch.Generator(device=device).manual_seed(seed)
    model.eval()

    # Save model state for restoration after sampling.
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

    tau_max = saved_tau_max
    if tau_max <= 0:
        tau_max = 1.0

    # Reverse trajectory: tau_max -> 0 (alpha_bar 0 -> 1)
    taus = torch.linspace(tau_max, 1e-6, n_steps, device=device)

    # Initialise Q in the U_q spectral subspace.
    U_q = model.U_q  # (F, q) geometric prior eigenbasis
    z = torch.randn(n_samples, U_q.shape[1], device=device, generator=gen)
    Q = z @ U_q.t()  # (n_samples, F) -- lifted to item space
    Q = F.normalize(Q, dim=-1)  # match L2-normalised data

    for i in range(n_steps - 1):
        tau_cur = float(taus[i])
        tau_next = float(taus[i + 1])

        # Set clock to current tau so alpha_bar is correct.
        clock.tau_n.fill_(tau_cur)
        clock.tau_max_est.fill_(tau_max)

        ab_cur = float(clock.alpha_bar().item())
        ab_next = float(clock.alpha_bar(tau=torch.tensor(tau_next, device=device)).item())

        # Clamp for numerical stability.
        ab_cur = max(ab_cur, 1e-8)
        ab_next = max(ab_next, 1e-8)

        # Step a: decode Q to x_hat using the diffusion decoder formula.
        # x_hat = sqrt(alpha_bar) * Q (noise=0, as in training forward pass)
        x_hat = (ab_cur ** 0.5) * Q

        # Step b: L2-normalise x_hat so the model sees data-distribution input.
        x_hat_norm = F.normalize(x_hat, dim=-1)

        # Step c: run model forward pass to get Q_predicted.
        torch.manual_seed(seed + i)
        with torch.no_grad():
            _ = model(x_hat_norm)
            Q_predicted = model._last_Q  # (B, F) -- model's clean Q prediction

        # Normalise Q_predicted to match the data manifold.
        Q_predicted = F.normalize(Q_predicted, dim=-1)

        # Step d: DDIM-style update.
        # Q_next = sqrt(ab_next) * Q_predicted + sqrt(1 - ab_next) * noise
        # Noise constrained to U_q subspace (geometrically consistent with init).
        z_noise = torch.randn(n_samples, U_q.shape[1], device=device, generator=gen)
        noise = z_noise @ U_q.t()  # (n_samples, F) -- lifted to item space
        noise = F.normalize(noise, dim=-1)
        a_sqrt_next = ab_next ** 0.5
        noise_sqrt_next = (1.0 - ab_next) ** 0.5
        Q = a_sqrt_next * Q_predicted + noise_sqrt_next * noise

    # Final decode: at alpha_bar=1, x_hat = Q.
    result = F.normalize(Q, dim=-1)

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
