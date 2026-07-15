"""
ESDM training CLI.

Usage:
    esdm-train --config esdm/_configs/default.yaml [overrides...]
    esdm-train --dataset mnist --steps 1000
    esdm-train --dataset cifar10 --lam-s 5.0

The --dataset flag selects the real-world data loader:
    random     -- synthetic Gaussian placeholder (legacy smoke path)
    mnist      -- MNIST PCA-16 embeddings
    cifar10    -- CIFAR-10 DINO-256 embeddings
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from esdm.data.cifar10 import CIFAR10DINOLoader, DEFAULT_OUTPUT_DIM as CIFAR_DIM
from esdm.data.mnist import MNISTLoader, DEFAULT_PCA_DIM as MNIST_DIM
from esdm.models.esdm import ESDM

logger = logging.getLogger("esdm.cli.train")

DATASET_CHOICES = ("random", "mnist", "cifar10")


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def _build_loader(dataset: str, seed: int) -> tuple[object, int]:
    """Build a data loader for the requested dataset and return its input dimension."""
    if dataset == "mnist":
        loader = MNISTLoader(pca_dim=MNIST_DIM, seed=seed)
        return loader, MNIST_DIM
    if dataset == "cifar10":
        loader = CIFAR10DINOLoader(output_dim=CIFAR_DIM, seed=seed)
        return loader, CIFAR_DIM
    if dataset == "random":
        return None, None
    raise ValueError(f"Unknown dataset: {dataset}")


def _build_model(cfg: dict, args: argparse.Namespace) -> tuple[ESDM, object]:
    """Construct ESDM from config, applying CLI overrides for dataset."""
    loader, dim_input = _build_loader(args.dataset, args.seed)
    if dim_input is not None:
        cfg["model"]["dim_input"] = dim_input
        cfg["model"]["n_nodes"] = dim_input
        cfg["model"]["dim_latent"] = min(cfg["model"].get("dim_latent", dim_input), dim_input // 2)
        cfg["model"]["n_modes"] = min(cfg["model"].get("n_modes", dim_input), dim_input // 2)
    cfg["model"]["lam_s"] = args.lam_s
    return ESDM(cfg), loader


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-train",
        description="Train the Entropic Semantic Diffusion Model.",
    )
    parser.add_argument("--config", "-c", default="esdm/_configs/default.yaml")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="random")
    parser.add_argument("--lam-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps from config")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = _load_config(args.config)
    if args.steps is not None:
        cfg["training"]["max_steps"] = args.steps

    logger.info("Building ESDM with dataset=%s, lam_s=%g", args.dataset, args.lam_s)
    model, loader = _build_model(cfg, args)
    logger.info("Model instantiated. dim_input=%d, n_nodes=%d", model.dim_input, model.n_nodes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    max_steps = cfg["training"]["max_steps"]
    batch_size = cfg["training"]["batch_size"]

    logger.info("Starting training: %d steps, batch_size=%d", max_steps, batch_size)
    for step in range(max_steps):
        if loader is not None:
            x = loader.train_batch(batch_size, seed=args.seed + step)
        else:
            x = torch.randn(batch_size, model.dim_input)
        optimizer.zero_grad()
        out = model(x)
        out.loss.backward()
        optimizer.step()
        if step % cfg["training"]["log_every"] == 0:
            hc = model.health_check()
            logger.info(
                "step=%d  loss=%.4f  tau_n=%.4f  alpha_bar=%.4f  heat_death=%s",
                step, float(out.loss), hc["tau_n"],
                float(model.clock.alpha_bar()), hc["heat_death"],
            )
        if hc["heat_death"]:
            logger.info("Heat death detected at step %d. Stopping.", step)
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
