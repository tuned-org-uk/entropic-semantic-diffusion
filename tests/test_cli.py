"""
Tests for the ESDM CLI: --wht-filter {none,basic,bivariate} flag.
"""
import numpy as np
import torch

from esdm.cli.wht_filter import build_wht_filter, WHT_FILTER_CHOICES
from esdm.cli.train import _build_model, _load_config
from esdm.graphs.wht_filter import GraphWHTFilter


def _make_L_F(F=16):
    rng = np.random.default_rng(3407)
    A = rng.standard_normal((F, F))
    L = A.T @ A
    np.fill_diagonal(L, L.diagonal() + 1e-3)
    return (L + L.T) / 2.0


def test_wht_filter_choices():
    assert "none" in WHT_FILTER_CHOICES
    assert "basic" in WHT_FILTER_CHOICES
    assert "bivariate" in WHT_FILTER_CHOICES


def test_build_wht_filter_none_returns_none():
    L_F = _make_L_F()
    assert build_wht_filter("none", L_F) is None


def test_build_wht_filter_basic():
    L_F = _make_L_F()
    filt = build_wht_filter("basic", L_F)
    assert filt is not None
    assert isinstance(filt, GraphWHTFilter)


def test_build_wht_filter_bivariate():
    L_F = _make_L_F()
    filt = build_wht_filter("bivariate", L_F, a_tau_scale=7.0, tau_decay_power=2.0)
    assert filt is not None
    assert isinstance(filt, GraphWHTFilter)
    assert filt.a_tau_scale == 7.0
    assert filt.tau_decay_power == 2.0


def test_build_model_with_none_filter():
    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8

    class FakeArgs:
        wht_filter = "none"
        a_tau_scale = 5.0
        tau_decay_power = 1.0
        seed = 3407

    model = _build_model(cfg, FakeArgs())
    assert model.wht_filter is None


def test_build_model_with_basic_filter():
    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8

    class FakeArgs:
        wht_filter = "basic"
        a_tau_scale = 5.0
        tau_decay_power = 1.0
        seed = 3407

    model = _build_model(cfg, FakeArgs())
    assert model.wht_filter is not None
    assert isinstance(model.wht_filter, GraphWHTFilter)


def test_build_model_with_bivariate_filter():
    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8

    class FakeArgs:
        wht_filter = "bivariate"
        a_tau_scale = 5.0
        tau_decay_power = 1.0
        seed = 3407

    model = _build_model(cfg, FakeArgs())
    assert model.wht_filter is not None
    assert model.wht_filter.a_tau_scale == 5.0


def test_forward_with_none_vs_basic_both_finite():
    """Both --wht-filter none and basic must produce finite forward passes."""
    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8

    torch.manual_seed(3407)
    x = torch.randn(4, 32)

    class FakeArgsNone:
        wht_filter = "none"
        a_tau_scale = 5.0
        tau_decay_power = 1.0
        seed = 3407

    class FakeArgsBasic:
        wht_filter = "basic"
        a_tau_scale = 5.0
        tau_decay_power = 1.0
        seed = 3407

    cfg_none = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    model_none = _build_model(cfg_none, FakeArgsNone())
    out_none = model_none(x)
    assert torch.isfinite(out_none.loss)

    cfg_basic = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    model_basic = _build_model(cfg_basic, FakeArgsBasic())
    out_basic = model_basic(x)
    assert torch.isfinite(out_basic.loss)
