"""Input-space ancestral sampler for the entropic-clock ESDM (issue #23)."""
from __future__ import annotations

import torch
from torch import Tensor

from ._clock_sampling import chart_noise, reverse_trajectory


def entropic_reverse_sde(
    model,
    n_samples: int = 64,
    n_steps: int = 100,
    seed: int = 3407,
    device: torch.device | None = None,
) -> Tensor:
    """Generate data coordinates with the model's clean signal and the clock schedule.

    The initial draw lies in the frozen chart. Later ancestral noise is
    Gaussian in input space; neither distribution is projected to unit length.
    A one-point grid returns the prior without a reverse transition.
    """
    chart = model.U_q
    gen = torch.Generator(device=chart.device).manual_seed(seed)
    with reverse_trajectory(model, n_steps, device) as taus:
        clock = model.clock
        x = chart_noise(chart, n_samples, gen)
        for current, following in zip(taus[:-1], taus[1:]):
            clock.tau_n.copy_(current)
            clock.tau_max_est.copy_(taus[0])
            a = clock.alpha_bar()
            a_next = clock.alpha_bar(tau=following)
            model(x, generator=gen)
            # The ancestral formula needs the clean estimate, before schedule
            # attenuation. It already applies the schedule to this estimate below.
            x_clean = model.readout(model._last_Q, chart)
            residual_sd = (1 - a).sqrt()
            direction = (x - a.sqrt() * x_clean) / residual_sd.clamp_min(clock.eps)
            mean = a_next.sqrt() * x_clean + (1 - a_next).sqrt() * direction

            pump = clock.Lambda_tau
            pump = torch.where(pump > 0, pump, 1.0)
            variance = ((1 - a_next) / (1 - a).clamp_min(clock.eps)
                        * (1 - pump / clock.lambda_clip)).clamp(0, 1)
            sigma = (variance * (current - following)).sqrt()
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
            x = mean + sigma * noise
        return x
