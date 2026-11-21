"""
Phase 6 test — ESDM top-level model.
Smoke-tests only until all sub-modules are implemented.
"""
import yaml
import pytest
import torch
from pathlib import Path


def _default_cfg():
    p = Path("esdm/_configs/default.yaml")
    with p.open() as f:
        return yaml.safe_load(f)


def test_esdm_instantiates():
    from esdm.models.esdm import ESDM
    model = ESDM(_default_cfg())
    assert model is not None


def test_health_check_runs():
    from esdm.models.esdm import ESDM
    model = ESDM(_default_cfg())
    hc = model.health_check()
    assert "tau_n" in hc
    assert "heat_death" in hc


def test_forward_pass():
    import torch
    from esdm.models.esdm import ESDM
    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    model = ESDM(cfg)
    x = torch.randn(4, cfg["model"]["dim_input"])
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert torch.isfinite(out.loss)
    assert out.clock is not None


@pytest.mark.parametrize("q", [1, 3])
def test_zero_time_forward_preserves_raw_wave_and_decodes_in_orthonormal_chart(q):
    from esdm.models import ESDM

    torch.manual_seed(3407)
    cfg = _default_cfg()
    cfg["model"].update(dim_input=6, n_nodes=6, dim_latent=q, n_modes=q)
    model = ESDM(cfg).double()
    chart = model.U_q
    torch.testing.assert_close(chart.T @ chart, torch.eye(q, dtype=chart.dtype))
    x = 10 * torch.randn(4, 6, dtype=chart.dtype)
    out = model(x)
    assert out.clock.delta_tau == 0
    torch.testing.assert_close(model._last_Q, x)
    signal = x + ((x @ chart) * (model.readout.weights() - 1)) @ chart.T
    torch.testing.assert_close(out.x_hat, signal)
    out.loss.backward()
    assert model.readout.logits.grad.abs().min() > 0


def test_encode_preserves_reparameterised_gaussian_draw():
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=4, n_nodes=4, dim_latent=2, n_modes=2)
    model = ESDM(cfg)
    x = torch.ones(4, 4)
    with torch.no_grad():
        model.enc_mu.weight.zero_()
        model.enc_mu.bias.fill_(10.0)
        model.enc_logvar.weight.zero_()
        model.enc_logvar.bias.fill_(2.0)
    torch.manual_seed(3407)
    mu, logvar, z = model.encode(x)
    eps = torch.randn(z.shape, generator=torch.Generator().manual_seed(3407))
    torch.testing.assert_close(z, mu + (logvar / 2).exp() * eps)
    assert (z.norm(dim=-1) > 1).all()


def test_model_rejects_incompatible_wave_and_chart_dimensions():
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=4, n_nodes=5, dim_latent=2, n_modes=2)
    with pytest.raises(ValueError, match="n_nodes"):
        ESDM(cfg)


@pytest.mark.parametrize("q", [0, 5])
def test_model_rejects_invalid_chart_rank_before_graph_processing(q):
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=4, n_nodes=4, dim_latent=q, n_modes=q)
    with pytest.raises(ValueError, match="1 <= dim_latent <= n_nodes"):
        ESDM(cfg)


def test_trained_state_round_trip_resumes_clock_and_bounded_signal():
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=8, n_nodes=8, dim_latent=2, n_modes=2)
    torch.manual_seed(3407)
    trained = ESDM(cfg).eval()
    opt = torch.optim.SGD(trained.parameters(), lr=0.01)
    x = torch.randn(3, 8)
    trained(x).loss.backward()
    opt.step()
    trained(x)

    restored = ESDM(cfg).eval().double()
    restored.load_state_dict(trained.state_dict())
    trained.double()
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, msg=key)
    a = trained(x.double(), generator=torch.Generator().manual_seed(3407))
    b = restored(x.double(), generator=torch.Generator().manual_seed(3407))
    torch.testing.assert_close(a.x_hat, b.x_hat)
    assert a.clock == b.clock


def test_model_load_rejects_legacy_or_invalid_frozen_chart():
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=8, n_nodes=8, dim_latent=2, n_modes=2)
    model = ESDM(cfg)
    state = model.state_dict()
    state["U_q"] = 2 * state["U_q"]
    with pytest.raises(ValueError, match="orthonormal"):
        model.load_state_dict(state)
    del state["readout.logits"]
    with pytest.raises(ValueError, match="readout"):
        model.load_state_dict(state, strict=False)


def test_configured_graph_and_gains_follow_model_default_dtype():
    from esdm.models import ESDM

    cfg = _default_cfg()
    cfg["model"].update(dim_input=4, n_nodes=4, dim_latent=2, n_modes=2, L_base=torch.eye(4))
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = ESDM(cfg)
        x = torch.ones(2, 4)
        torch.testing.assert_close(model(x).x_hat, model.readout(x, model.U_q))
    finally:
        torch.set_default_dtype(default)





def test_training_loop_advances_entropic_time():
    """A short training loop must increase tau_n and decrease alpha_bar."""
    import torch
    from esdm.models.esdm import ESDM

    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    cfg["model"]["n_vibrational_steps"] = 2

    torch.manual_seed(42)
    model = ESDM(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    taus = []
    alphas = []
    for step in range(100):
        x = torch.randn(8, cfg["model"]["dim_input"])
        optimizer.zero_grad()
        out = model(x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        taus.append(out.clock.tau_n)
        alphas.append(float(model.clock.alpha_bar()))

    # Entropic time must be strictly increasing across the loop.
    assert all(taus[i + 1] > taus[i] for i in range(len(taus) - 1))
    # Diffusion schedule must decrease as tau accumulates.
    assert alphas[-1] < alphas[0]
    # Loss stayed finite.
    assert all(torch.isfinite(torch.tensor(a)) for a in alphas)


def test_lambdas_buffer_none_by_default():
    """lambdas buffer is None when not provided in config."""
    from esdm.models.esdm import ESDM
    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    model = ESDM(cfg)
    assert model.lambdas is None


def test_lambdas_buffer_stored_when_provided():
    """lambdas buffer is stored when provided in config."""
    import torch
    from esdm.models.esdm import ESDM
    cfg = _default_cfg()
    cfg["model"]["dim_input"] = 32
    cfg["model"]["dim_latent"] = 8
    cfg["model"]["n_nodes"] = 32
    cfg["model"]["n_modes"] = 8
    cfg["model"]["L_base"] = torch.eye(32)
    cfg["model"]["lambdas"] = torch.rand(100)
    model = ESDM(cfg)
    assert model.lambdas is not None
    assert model.lambdas.shape == (100,)
    assert float(model.lambdas.min()) >= 0.0
    assert float(model.lambdas.max()) <= 1.0
