"""The mean starts in graph coordinates without clipping Gaussian samples."""

import torch

from esdm.experiments.v2_readout import _model, path_data


def test_initial_mean_recovers_chart_coefficients_and_remains_trainable():
    model = _model(path_data(), 8)
    coefficients = torch.arange(1, 9, dtype=model.U_q.dtype).reshape(1, -1) * 100
    x = coefficients @ model.U_q.T
    mean, _, sample = model.encode(x, generator=torch.Generator().manual_seed(3407))
    torch.testing.assert_close(mean, coefficients)
    assert sample.norm() > 100
    mean.square().sum().backward()
    assert model.enc_mu.weight.grad.norm() > 0
