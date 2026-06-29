"""
Phase 2 test — EntropicClock.
All assertions from 01-IMPLEMENTATION-PLAN.md §3.4.
"""
import pytest
import torch


@pytest.mark.skip(reason="Phase 2 not yet implemented")
def test_tau_n_monotone():
    from esdm.clock import EntropicClock
    clock = EntropicClock(eps=1e-8, heat_death_patience=200)
    torch.manual_seed(42)
    tau_prev = 0.0
    for _ in range(1000):
        kl = torch.rand(1).squeeze()
        phi = torch.randn(1).squeeze()
        rho_p = torch.rand(1).squeeze()
        rho_m = torch.rand(1).squeeze()
        dark = torch.rand(1).squeeze()
        state = clock.step(kl, phi, rho_p, rho_m, dark)
        assert state.tau_n >= tau_prev - 1e-9
        tau_prev = state.tau_n


@pytest.mark.skip(reason="Phase 2 not yet implemented")
def test_alpha_bar_bounds():
    from esdm.clock import EntropicClock
    clock = EntropicClock()
    clock.tau_n.fill_(0.0)
    clock.tau_max_est.fill_(100.0)
    ab_start = clock.alpha_bar(torch.tensor(0.0))
    ab_end   = clock.alpha_bar(torch.tensor(100.0))
    assert abs(float(ab_start) - 1.0) < 1e-5
    assert abs(float(ab_end)   - 0.0) < 1e-5


@pytest.mark.skip(reason="Phase 2 not yet implemented")
def test_heat_death_triggers():
    from esdm.clock import EntropicClock
    patience = 10
    clock = EntropicClock(heat_death_patience=patience)
    clock._flat_steps.fill_(patience)
    assert clock.is_heat_death()


@pytest.mark.skip(reason="Phase 2 not yet implemented")
def test_lambda_clipped():
    from esdm.clock import EntropicClock
    clock = EntropicClock(lambda_clip=100.0)
    # Very small d_phi -> Lambda would explode without clipping
    kl = torch.tensor(1.0)
    phi = torch.tensor(1e-15)
    state = clock.step(kl, phi,
                       torch.tensor(0.5), torch.tensor(0.5),
                       torch.tensor(0.1))
    assert state.Lambda_tau <= 100.0 + 1e-6
