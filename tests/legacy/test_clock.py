"""
Phase 2 test — EntropicClock.
All assertions from 01-IMPLEMENTATION-PLAN.md §3.4.
"""
import pytest
import torch


def test_tau_n_monotone():
    from esdm_legacy.clock import EntropicClock
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


def test_alpha_bar_bounds():
    from esdm_legacy.clock import EntropicClock
    clock = EntropicClock()
    clock.tau_n.fill_(0.0)
    clock.tau_max_est.fill_(100.0)
    ab_start = clock.alpha_bar(torch.tensor(0.0))
    ab_end   = clock.alpha_bar(torch.tensor(100.0))
    assert abs(float(ab_start) - 1.0) < 1e-5
    assert abs(float(ab_end)   - 0.0) < 1e-5


def test_heat_death_triggers():
    from esdm_legacy.clock import EntropicClock
    patience = 10
    clock = EntropicClock(heat_death_patience=patience)
    clock._flat_steps.fill_(patience)
    assert clock.is_heat_death()


def test_lambda_clipped():
    """Lambda_tau is clipped when d_kl is near zero (stable mode)."""
    from esdm_legacy.clock import EntropicClock
    clock = EntropicClock(lambda_clip=100.0, eps=1e-8)
    # First step establishes prev state.
    clock.step(
        kl_tau_current=torch.tensor(1.0),
        phi_current=torch.tensor(0.0),
        rho_plus=torch.tensor(0.5),
        rho_minus=torch.tensor(0.5),
        dark_entropy=torch.tensor(0.1),
    )
    # Stable mode: kl_tau unchanged (d_kl = 0) while phi moves freely.
    # denom -> eps, Lambda -> 1/eps, which must be clipped to lambda_clip.
    state = clock.step(
        kl_tau_current=torch.tensor(1.0),
        phi_current=torch.tensor(1.0),
        rho_plus=torch.tensor(0.5),
        rho_minus=torch.tensor(0.5),
        dark_entropy=torch.tensor(0.1),
    )
    assert state.Lambda_tau == pytest.approx(100.0, abs=1e-4)


def test_lambda_small_when_entropy_flows():
    """Lambda_tau is small when entropy flows rapidly (large d_kl / d_phi)."""
    from esdm_legacy.clock import EntropicClock
    clock = EntropicClock(lambda_clip=100.0, eps=1e-8)
    clock.step(
        kl_tau_current=torch.tensor(0.0),
        phi_current=torch.tensor(0.0),
        rho_plus=torch.tensor(0.5),
        rho_minus=torch.tensor(0.5),
        dark_entropy=torch.tensor(0.1),
    )
    state = clock.step(
        kl_tau_current=torch.tensor(1.0),
        phi_current=torch.tensor(0.1),
        rho_plus=torch.tensor(0.5),
        rho_minus=torch.tensor(0.5),
        dark_entropy=torch.tensor(0.1),
    )
    # d_kl = 1.0, d_phi = 0.1 -> denom = 10 -> Lambda = 0.1
    assert state.Lambda_tau < 1.0


def test_per_mode_tau_accumulates_delta():
    """per_mode_tau accumulates |Delta kl_tau|, not |kl_tau|."""
    from esdm_legacy.clock import EntropicClock
    clock = EntropicClock()
    sequence = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 3.0]),
    ]
    for kl in sequence:
        clock.step(
            kl_tau_current=kl,
            phi_current=torch.tensor(0.0),
            rho_plus=torch.tensor(0.5),
            rho_minus=torch.tensor(0.5),
            dark_entropy=torch.tensor(0.1),
        )
    per_mode = clock.per_mode_tau
    assert per_mode is not None
    # |Delta| sequence: [1,2], [1,2], [2,3] -> cumulative [4,7]
    assert torch.allclose(per_mode, torch.tensor([4.0, 7.0]), atol=1e-6)
