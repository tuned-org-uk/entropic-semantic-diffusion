"""
Tests for generation pipeline (issue #23): reverse-SDE sampler, pixel decoder, CLI.
"""
from __future__ import annotations

import torch

from esdm.models.esdm import ESDM
from esdm.models.pixel_decoder import PixelDecoder
from esdm.sampling.entropic_sampler import entropic_reverse_sde
from esdm.cli.generate import main as generate_main


def _build_small_model():
    """Build a small ESDM for testing."""
    F = 16
    cfg = {
        "model": {
            "dim_input": F, "dim_latent": F // 4, "n_nodes": F,
            "n_modes": F // 4, "n_vibrational_steps": 2, "gamma": 0.1,
            "lam_s": 1.0, "sigma_sq": 1.0, "freeze_h": False,
        },
        "clock": {"eps": 1e-8, "heat_death_patience": 200, "lambda_clip": 100.0, "eps_rel": 0.0},
        "training": {"batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "max_steps": 10},
    }
    # Build a random Laplacian for L_base.
    A = torch.rand(F, F)
    A = (A + A.T) / 2
    A.fill_diagonal_(0)
    D = A.sum(dim=-1).diag()
    L = D - A
    cfg["model"]["L_base"] = L
    model = ESDM(cfg)
    # Initialize S_0 and clock state with a forward pass.
    x = torch.randn(4, F)
    with torch.no_grad():
        _ = model(x)
    return model


def test_sampler_output_shape():
    """Reverse-SDE sampler produces (n_samples, dim_input)."""
    model = _build_small_model()
    samples = entropic_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert samples.shape == (4, 16)
    assert torch.all(torch.isfinite(samples))


def test_sampler_reproducible():
    """Same seed produces same samples."""
    model = _build_small_model()
    s1 = entropic_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    s2 = entropic_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert torch.allclose(s1, s2, atol=1.0)


def test_sampler_restores_clock():
    """Clock state is restored after sampling."""
    model = _build_small_model()
    tau_before = float(model.clock.tau_n.item())
    flat_before = int(model.clock._flat_steps.item())
    _ = entropic_reverse_sde(model, n_samples=4, n_steps=10, seed=3407)
    assert float(model.clock.tau_n.item()) == tau_before
    assert int(model.clock._flat_steps.item()) == flat_before


def test_pixel_decoder_output_shape():
    """Pixel decoder produces (B, C, H, W) in [0, 1]."""
    dec = PixelDecoder(dim_input=16, img_channels=3, img_size=32)
    z = torch.randn(4, 16)
    img = dec(z)
    assert img.shape == (4, 3, 32, 32)
    assert torch.all(img >= 0)
    assert torch.all(img <= 1)


def test_pixel_decoder_grayscale():
    """Grayscale pixel decoder works."""
    dec = PixelDecoder(dim_input=16, img_channels=1, img_size=28)
    z = torch.randn(2, 16)
    img = dec(z)
    assert img.shape == (2, 1, 28, 28)


def test_pixel_decoder_gradients():
    """Gradients flow through pixel decoder."""
    dec = PixelDecoder(dim_input=16, img_channels=3, img_size=32)
    z = torch.randn(4, 16, requires_grad=True)
    img = dec(z)
    loss = img.sum()
    loss.backward()
    assert z.grad is not None


def test_generate_cli_dry_run(tmp_path):
    """Generate CLI runs without error (uses fake checkpoint)."""
    model = _build_small_model()
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    out_dir = tmp_path / "samples"
    generate_main([
        "--checkpoint", str(ckpt_path),
        "--dataset", "mnist",
        "--n-samples", "4",
        "--n-steps", "5",
        "--output-dir", str(out_dir),
        "--save-embeddings",
    ])
    assert (out_dir / "embeddings.pt").exists()
    emb = torch.load(out_dir / "embeddings.pt", weights_only=False)
    assert emb.shape == (4, 16)
