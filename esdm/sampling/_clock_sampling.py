"""Shared chart prior and temporary state for the two entropic-clock samplers."""
from contextlib import contextmanager

import torch


def chart_noise(U_q, n_samples, generator):
    """Lift a standard Gaussian draw; the chart restricts direction, not length."""
    z = torch.randn(n_samples, U_q.shape[1], device=U_q.device, dtype=U_q.dtype,
                    generator=generator)
    return z @ U_q.T


@contextmanager
def reverse_trajectory(model, n_steps, device):
    """Evaluate a reverse grid without changing the training clock or wave history."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if device is not None and torch.device(device) != model.U_q.device:
        raise ValueError("sampling device must match the model's frozen chart")

    clock = model.clock
    wave = model.vibrational
    density = wave._vdt_block.density
    # Forward replaces some buffers (including initially-None buffers) and
    # writes others in place. Preserve originals; only working copies may change.
    names = [(model, name) for name in model._buffers if name == "S_0" or name.startswith("_last_")]
    names += [(module, name) for module in (clock, wave) for name in module._buffers]
    buffers = [(module, name, getattr(module, name)) for module, name in names]
    training = [(module, module.training) for module in model.modules()]
    # The vendored density update writes an EMA directly to two parameters and
    # stores two gradient bridge tensors outside its state_dict.
    params = [(p, p.detach().clone()) for p in density.parameters()]
    bridges = {k: v for k, v in vars(density).items() if k.startswith("_grad_bridge_")}

    with torch.no_grad():
        try:
            for module, name, value in buffers:
                if value is not None:
                    setattr(module, name, value.detach().clone())
            model.eval()
            horizon = clock.tau_max_est.item()
            if horizon <= 0:
                raise ValueError("a positive entropic-time horizon is required for sampling")
            yield torch.linspace(horizon, 0, n_steps, device=model.U_q.device,
                                 dtype=model.U_q.dtype)
        finally:
            for module, name, value in buffers:
                setattr(module, name, value)
            for p, value in params:
                p.copy_(value)
            for name in ("_grad_bridge_plus", "_grad_bridge_minus"):
                if name in bridges:
                    setattr(density, name, bridges[name])
                elif hasattr(density, name):
                    delattr(density, name)
            for module, train in training:
                module.training = train
