"""
Phase 6 test — ESDM top-level model.
Smoke-tests only until all sub-modules are implemented.
"""
import pytest
import yaml
from pathlib import Path


def _default_cfg():
    p = Path("esdm/_configs/default.yaml")
    with p.open() as f:
        return yaml.safe_load(f)


def test_esdm_instantiates():
    from esdm.model import ESDM
    model = ESDM(_default_cfg())
    assert model is not None


def test_health_check_runs():
    from esdm.model import ESDM
    model = ESDM(_default_cfg())
    hc = model.health_check()
    assert "tau_n" in hc
    assert "heat_death" in hc


def test_forward_pass():
    import torch
    from esdm.model import ESDM
    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_modes"] = 8
    model = ESDM(cfg)
    x = torch.randn(4, cfg["model"]["dim_input"])
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert torch.isfinite(out.loss)
    assert out.clock is not None
