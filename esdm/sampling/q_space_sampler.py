"""Reverse diffusion in the entropic-clock model's raw wave coordinates."""
from __future__ import annotations

import torch
from torch import Tensor

from ._clock_sampling import chart_noise, reverse_trajectory


def q_space_reverse_sde(
    model,
    n_samples: int = 64,
    n_steps: int = 100,
    seed: int = 3407,
    device: torch.device | None = None,
) -> Tensor:
    """Return generated embeddings using the trained model as a Q-space predictor.

    At each grid point decode the current wave to its deterministic signal.
    Evolve this input with the model and mix its raw wave prediction with fresh
    chart Gaussian noise at the next time. Only the final readout maps that
    wave back into data space. A one-point grid reads out the prior directly.
    """
    chart = model.U_q
    gen = torch.Generator(device=chart.device).manual_seed(seed)
    with reverse_trajectory(model, n_steps, device) as taus:
        clock = model.clock
        Q = chart_noise(chart, n_samples, gen)
        for current, following in zip(taus[:-1], taus[1:]):
            clock.tau_n.copy_(current)
            clock.tau_max_est.copy_(taus[0])
            a = clock.alpha_bar()
            a_next = clock.alpha_bar(tau=following)
            signal = a.sqrt() * model.readout(Q, chart)
            model(signal, generator=gen)
            noise = chart_noise(chart, n_samples, gen)
            Q = a_next.sqrt() * model._last_Q + (1 - a_next).sqrt() * noise
        return model.readout(Q, chart)
