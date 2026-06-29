"""
Phase 3 test — VibrationalStateBlock.
All assertions from 01-IMPLEMENTATION-PLAN.md §4.4.
"""
import pytest
import torch

from esdm.vibrational import VibrationalStateBlock


def test_vibrational_instantiates():
    block = VibrationalStateBlock(dim=16, n_heads=2, n_steps=3)
    assert block is not None


def test_vibrational_forward_is_stub():
    block = VibrationalStateBlock(dim=16)
    x = torch.randn(2, 16)
    L_f = torch.randn(16, 16)
    with pytest.raises(NotImplementedError):
        block(x, L_f, Lambda_tau=1.0, delta_tau=0.1)


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_vibrational_matches_vdt_when_lambda_one():
    """With Lambda_tau=1.0 and delta_tau=dt, output matches original VDT."""
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_vibrational_forcing_vanishes_when_lambda_zero():
    """With Lambda_tau=0.0, the forcing term B_t vanishes from the recurrence."""
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3 not yet implemented")
def test_vibrational_output_finite_for_random_inputs():
    """Q_next is finite and bounded for random inputs."""
    raise NotImplementedError
