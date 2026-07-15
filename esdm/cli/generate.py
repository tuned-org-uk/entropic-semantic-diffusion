"""
ESDM generation CLI (issue #23).

Usage:
    esdm-generate --checkpoint model.pt --n-samples 64 --output-dir samples/
    esdm-generate --checkpoint model.pt --dataset cifar10 --n-samples 16 --n-steps 50

Loads a trained ESDM checkpoint, runs the reverse-SDE sampler, optionally
decodes embeddings to images, and saves them.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from esdm.models.esdm import ESDM
from esdm.models.pixel_decoder import PixelDecoder
from esdm.sampling.entropic_sampler import entropic_reverse_sde

logger = logging.getLogger("esdm.cli.generate")


def _load_config(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="esdm-generate",
        description="Generate samples from a trained ESDM.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="esdm/_configs/default.yaml", help="Model config YAML")
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=100, help="Reverse-SDE steps")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default="samples/", help="Output directory")
    parser.add_argument("--save-embeddings", action="store_true", help="Save raw embeddings as .pt")
    parser.add_argument("--pixel-decoder-checkpoint", default=None, help="Path to pixel decoder .pt")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model config and set dataset dims.
    cfg = _load_config(args.config)
    if args.dataset == "cifar10":
        from esdm.data.cifar10 import DEFAULT_OUTPUT_DIM
        dim = DEFAULT_OUTPUT_DIM
        img_channels, img_size = 3, 32
    else:
        from esdm.data.mnist import DEFAULT_PCA_DIM
        dim = DEFAULT_PCA_DIM
        img_channels, img_size = 1, 28

    cfg["model"]["dim_input"] = dim
    cfg["model"]["n_nodes"] = dim
    q = max(1, dim // 4)
    cfg["model"]["dim_latent"] = q
    cfg["model"]["n_modes"] = q

    model = ESDM(cfg)

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Filter out runtime buffers that depend on batch size / training state.
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    # Generate embeddings.
    logger.info("Generating %d samples with %d reverse-SDE steps...", args.n_samples, args.n_steps)
    embeddings = entropic_reverse_sde(
        model,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    logger.info("Generated embeddings shape: %s", tuple(embeddings.shape))

    # Save.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        torch.save(embeddings, out_dir / "embeddings.pt")
        logger.info("Saved embeddings to %s", out_dir / "embeddings.pt")

    # Optional: decode to pixels.
    if args.pixel_decoder_checkpoint:
        pd_ckpt = torch.load(args.pixel_decoder_checkpoint, map_location="cpu", weights_only=False)
        pixel_dec = PixelDecoder(dim_input=dim, img_channels=img_channels, img_size=img_size)
        if "pixel_decoder_state_dict" in pd_ckpt:
            pixel_dec.load_state_dict(pd_ckpt["pixel_decoder_state_dict"])
        else:
            pixel_dec.load_state_dict(pd_ckpt)
        pixel_dec.eval()
        with torch.no_grad():
            images = pixel_dec(embeddings)
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            if img_channels == 1:
                img = img[0]
            np.save(out_dir / f"sample_{i:04d}.npy", img)
        logger.info("Saved %d images to %s", images.shape[0], out_dir)
    else:
        logger.info("No pixel decoder checkpoint provided. Skipping image decode.")
        logger.info("To decode embeddings to images, train a PixelDecoder and pass --pixel-decoder-checkpoint.")

    logger.info("Generation complete.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
