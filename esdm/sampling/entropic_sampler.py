"""
Reverse-SDE sampler for the entropic-clock ESDM (issue #23).

Samples from a trained ESDM by reversing the entropic diffusion schedule:
  - Start from x ~ N(0, I) at tau = tau_max (alpha_bar = 0, full noise)
  - At each reverse step, run the model forward to predict the clean sample
  - Use the DDPM ancestral update rule to step from x_tau to x_{tau-dtau}
  - Stop when tau reaches 0 (alpha_bar = 1, clean sample)

The entropic pump Lambda(tau) modulates the reverse step size — larger
Lambda means smaller steps (more careful denoising in high-entropy-flow
regimes), matching the forward-process collocation strategy.

See docs/00-DESIGN.md §4.2 and 001-IMPLEMENTATION-PLAN.md §6.2.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def entropic_reverse_sde(
    model,
    n_samples: int = 64,
    n_steps: int = 100,
    seed: int = 3407,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Sample from a trained ESDM via reverse entropic diffusion.

    Parameters
    ----------
    model : ESDM
        Trained ESDM model. Must be in eval mode.
    n_samples : int
        Number of samples to generate.
    n_steps : int
        Number of reverse-SDE steps (more = higher quality, slower).
    seed : int
        Random seed for reproducibility.
    device : torch.device, optional

    Returns
    -------
    Tensor, shape (n_samples, dim_input)
        Generated embeddings in the data space.
    """
    if device is None:
        device = next(model.parameters()).device

    gen = torch.Generator(device=device).manual_seed(seed)
    model.eval()

    # Save clock and model state so we can restore after sampling.
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

    # Get the entropic time horizon from the trained clock.
    tau_max = saved_tau_max
    if tau_max <= 0:
        tau_max = 1.0

    # Reverse trajectory: tau_max -> 0
    taus = torch.linspace(tau_max, 1e-6, n_steps, device=device)

    # Start from noise constrained to the U_q spectral subspace (issue #29 Phase 1).
    # The forward pass projects Q through P = U_q @ U_q^T, so starting outside
    # that subspace means the first projection destroys most of the signal.
    # Q_init = U_q @ z where z ~ N(0, I_q), then L2-normalise to match data.
    U_q = model.U_q  # (F, q) geometric prior eigenbasis
    z = torch.randn(n_samples, U_q.shape[1], device=device, generator=gen)
    x = z @ U_q.t()  # (n_samples, F) -- lifted to item space
    x = torch.nn.functional.normalize(x, dim=-1)  # match L2-normalised data

    for i in range(n_steps - 1):
        tau_cur = float(taus[i])
        tau_next = float(taus[i + 1])
        dtau = tau_cur - tau_next  # positive (going backward)

        # Set clock to current tau so alpha_bar is correct.
        clock.tau_n.fill_(tau_cur)
        clock.tau_max_est.fill_(tau_max)

        # Forward pass: model predicts the clean sample from x.
        with torch.no_grad():
            torch.manual_seed(seed + i)  # deterministic forward
            out = model(x)
            x_0_pred = out.x_hat  # predicted clean embedding

        # Alpha_bar at current and next tau.
        ab_cur = float(clock.alpha_bar().item())
        ab_next = clock.alpha_bar(tau=torch.tensor(tau_next, device=device))
        ab_next = float(ab_next.item())

        # Clamp for numerical stability.
        ab_cur = max(ab_cur, 1e-8)
        ab_next = max(ab_next, 1e-8)

        # DDPM ancestral update:
        #   x_{tau-dtau} = sqrt(ab_next) * x_0_pred
        #                 + sqrt(1 - ab_next) * (x - sqrt(ab_cur) * x_0_pred) / sqrt(1 - ab_cur)
        #                 + sigma * noise
        # where sigma controls the stochasticity of the reverse step.
        # Lambda(tau) modulates step size: higher Lambda -> smaller sigma.
        lambda_tau = float(clock.Lambda_tau.item()) if clock.Lambda_tau.item() > 0 else 1.0
        sigma = torch.tensor(
            (1.0 - ab_next) / (1.0 - ab_cur + 1e-8) * (1.0 - lambda_tau / clock.lambda_clip),
            device=device,
        ).clamp(0.0, 1.0).sqrt() * (dtau ** 0.5)

        noise = torch.randn_like(x, generator=gen)

        # Deterministic part: move toward predicted clean sample.
        a_sqrt_next = ab_next ** 0.5
        a_sqrt_cur = ab_cur ** 0.5
        noise_sqrt_cur = (1.0 - ab_cur) ** 0.5
        noise_sqrt_next = (1.0 - ab_next) ** 0.5

        # Direction pointing to x_0 from x_t.
        mean = a_sqrt_next * x_0_pred + noise_sqrt_next * (x - a_sqrt_cur * x_0_pred) / (noise_sqrt_cur + 1e-8)

        x = mean + sigma * noise

    # Restore clock and model state.
    clock.tau_n.fill_(saved_tau_n)
    clock.tau_max_est.fill_(saved_tau_max)
    clock._prev_kl_tau.fill_(saved_prev_kl)
    clock._flat_steps.fill_(saved_flat)
    clock._prev_phi.fill_(saved_prev_phi)
    model._last_rho_plus = saved_rho_plus
    model._last_rho_minus = saved_rho_minus
    model._last_N_active.fill_(saved_N_active)
    model.vibrational.Q_tm1 = saved_Q_tm1

    return x
