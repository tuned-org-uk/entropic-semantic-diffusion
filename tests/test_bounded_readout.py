"""The v2 bound is on modal gains, not on the amplitude of a signal."""
import numpy as np
import pytest
import torch
from scipy.special import expi


@pytest.mark.parametrize("counts", [[564], [564, 1028, 1900, 3512, 6542]])
def test_gains_initialise_at_prime_counts_over_unoffset_li(counts):
    from esdm.models import BoundedSpectralReadout

    readout = BoundedSpectralReadout(len(counts))
    # For five modes the specified grid consists of the five successive doublings.
    x = 4096 * 2 ** np.arange(len(counts))
    li = expi(np.log(x))
    radius = np.sqrt(x) * np.log(x) / (8 * np.pi * li)
    np.testing.assert_allclose(readout.weights().detach().numpy(), counts / li, rtol=1e-7)
    np.testing.assert_allclose(readout.radius.numpy(), radius, rtol=1e-7)


def test_rounded_grid_counts_prime_at_its_upper_endpoint():
    from esdm.models import BoundedSpectralReadout

    # The fourth of ten log-spaced integers rounds to 10321, itself prime.
    # Trial division is an independent oracle for inclusion of the endpoint.
    from math import isqrt, log

    count = sum(all(n % p for p in range(2, isqrt(n) + 1)) for n in range(2, 10322))
    gain = BoundedSpectralReadout(10).weights()[3].item()
    assert gain == pytest.approx(count / expi(log(10321)), rel=1e-7)


def test_rotated_modes_gain_and_complement_survive_optimisation_and_restore():
    from esdm.models import BoundedSpectralReadout

    readout = BoundedSpectralReadout(2).double()
    chart, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=torch.Generator().manual_seed(3407), dtype=torch.float64)
    )
    basis = chart[:, :2]
    signals = 10 * chart.T
    initial_weights = readout.weights().detach().clone()
    opt = torch.optim.SGD(readout.parameters(), lr=1e6)
    # Drive opposite modes far past both ends of the allowed interval.
    coeffs = (readout(signals, basis) @ chart).diagonal()
    (coeffs[0] - coeffs[1]).backward()
    opt.step()
    restored = BoundedSpectralReadout(2).double()
    restored.load_state_dict(readout.state_dict())
    gains = restored.weights()
    assert gains[0] < initial_weights[0] and gains[1] > initial_weights[1]
    assert ((1 - restored.radius <= gains) & (gains <= 1 + restored.radius)).all()
    expected = torch.cat((gains, gains.new_ones(1)))[:, None] * signals
    torch.testing.assert_close(restored(signals, basis), expected)
    torch.testing.assert_close(restored(chart.T, basis), expected / 10)


def test_readout_rejects_chart_mode_broadcast():
    from esdm.models import BoundedSpectralReadout

    readout = BoundedSpectralReadout(1)
    with pytest.raises(ValueError, match="U_q"):
        readout(torch.ones(3, 4), torch.eye(4)[:, :2])
