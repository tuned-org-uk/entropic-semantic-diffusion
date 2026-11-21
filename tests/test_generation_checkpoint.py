"""A generation checkpoint must identify the v2 reconstruction it will run."""

import copy

import pytest
import torch

from esdm.cli.generate import _restore_generation_model, main
from esdm.models.esdm import ESDM


def _config():
    return {
        "model": {
            "dim_input": 8,
            "n_nodes": 8,
            "dim_latent": 2,
            "n_modes": 2,
            "gamma": 0.1,
            "n_vibrational_steps": 2,
        },
        "clock": {"eps": 1e-8, "heat_death_patience": 50},
    }


def test_generation_refuses_to_invent_missing_readout():
    cfg = _config()
    state = {k: v for k, v in ESDM(cfg).state_dict().items() if not k.startswith("readout.")}
    with pytest.raises(ValueError, match="readout"):
        _restore_generation_model(cfg, state)


def test_generation_restores_the_trained_operator_and_clock():
    torch.manual_seed(3407)
    cfg = _config()
    model = ESDM(cfg).eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        model(x)
        model.readout.logits.fill_(0.8)
    restored = _restore_generation_model(cfg, model.state_dict()).eval()
    torch.testing.assert_close(restored.readout.weights(), model.readout.weights())
    torch.testing.assert_close(restored.S_0, model.S_0)
    torch.testing.assert_close(restored.U_q, model.U_q)
    with torch.no_grad():
        torch.manual_seed(3407)
        expected = model(x).x_hat
        torch.manual_seed(3407)
        torch.testing.assert_close(restored(x).x_hat, expected)


def test_generation_rejects_nonorthonormal_frame():
    cfg = _config()
    state = copy.deepcopy(ESDM(cfg).state_dict())
    state["U_q"][:, 0] *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        _restore_generation_model(cfg, state)


def test_generation_uses_saved_chart_rank_for_bare_checkpoint(tmp_path):
    cfg = _config()
    cfg["model"].update(n_nodes=16, dim_input=16, dim_latent=3, n_modes=3)
    torch.manual_seed(3407)
    checkpoint = tmp_path / "rank-three.pt"
    torch.save(ESDM(cfg).state_dict(), checkpoint)
    main(["--checkpoint", str(checkpoint), "--dataset", "mnist", "--n-steps", "2",
          "--n-samples", "2", "--output-dir", str(tmp_path), "--save-embeddings"])
    assert torch.load(tmp_path / "embeddings.pt", weights_only=True).shape == (2, 16)
