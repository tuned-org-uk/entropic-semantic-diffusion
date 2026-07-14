"""
ESDM training CLI.

Usage:
    esdm-train --config esdm/_configs/default.yaml [overrides...]
    esdm-train --dataset mnist --steps 1000
    esdm-train --dataset cifar10 --wht-filter bivariate --a-tau-scale 5.0

The --dataset flag selects the real-world data loader:
    random     -- synthetic Gaussian placeholder (legacy smoke path)
    mnist      -- MNIST PCA-16 embeddings
    cifar10    -- CIFAR-10 DINO-256 embeddings

The --wht-filter flag selects the spectral filter directive:
    none       -- bare L_f @ Q (no WHT gating)
    basic      -- heat-kernel-only filter
    bivariate  -- Path A bivariate gate g_k(tau, r_k) with heat-death closure
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from esdm.cli.wht_filter import build_wht_filter, WHT_FILTER_CHOICES
from esdm.data.cifar10 import CIFAR10DINOLoader, DEFAULT_OUTPUT_DIM as CIFAR_DIM
from esdm.data.mnist import MNISTLoader, DEFAULT_PCA_DIM as MNIST_DIM
from esdm.models.esdm import ESDM

logger = logging.getLogger("esdm.cli.train")

#: Supported CLI datasets.
DATASET_CHOICES = ("random", "mnist", "cifar10")


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def _build_loader(dataset: str, seed: int) -> tuple[object, int]:
    """
    Build a data loader for the requested dataset and return its input dimension.

    Returns
    -------
    loader :
        Object with ``train_batch(batch_size, seed) -> Tensor``.
    dim_input : int
        Embedding dimension $F$ for the dataset.
    """
    if dataset == "mnist":
        loader = MNISTLoader(pca_dim=MNIST_DIM, seed=seed)
        return loader, MNIST_DIM
    if dataset == "cifar10":
        loader = CIFAR10DINOLoader(output_dim=CIFAR_DIM, seed=seed)
        return loader, CIFAR_DIM
    if dataset == "random":
        return None, None  # use config default
    raise ValueError(f"Unknown dataset: {dataset}")


def _build_model(cfg: dict, args: argparse.Namespace) -> ESDM:
    """Construct ESDM from config, applying CLI overrides for dataset and WHT filter."""
    loader, dim_input = _build_loader(args.dataset, args.seed)
    if dim_input is not None:
        cfg["model"]["dim_input"] = dim_input
        cfg["model"]["n_nodes"] = dim_input
        # Ensure latent dimensions do not exceed the graph node count.
        cfg["model"]["dim_latent"] = min(cfg["model"].get("dim_latent", dim_input), dim_input // 2)
        cfg["model"]["n_modes"] = min(cfg["model"].get("n_modes", dim_input), dim_input // 2)

    if args.wht_filter != "none":
        F = cfg["model"]["dim_input"]
        rng = np.random.default_rng(args.seed)
        A = rng.standard_normal((F, F))
        L_F = A.T @ A
        np.fill_diagonal(L_F, L_F.diagonal() + 1e-3)
        L_F = (L_F + L_F.T) / 2.0
        cfg["model"]["wht_filter"] = build_wht_filter(
            args.wht_filter,
            L_F,
            a_tau_scale=args.a_tau_scale,
            tau_decay_power=args.tau_decay_power,
        )
    else:
        cfg["model"]["wht_filter"] = None

    return ESDM(cfg), loader


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-train",
        description="Train the Entropic Semantic Diffusion Model.",
    )
    parser.add_argument(
        "--config", "-c",
        default="esdm/_configs/default.yaml",
        help="Path to YAML config file (default: esdm/_configs/default.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="random",
        help="Dataset loader to use (default: random)",
    )
    parser.add_argument(
        "--wht-filter",
        choices=WHT_FILTER_CHOICES,
        default="none",
        help="WHT spectral filter directive (default: none)",
    )
    parser.add_argument(
        "--a-tau-scale",
        type=float,
        default=5.0,
        help="Bivariate gate scale parameter (only used with --wht-filter bivariate)",
    )
    parser.add_argument(
        "--tau-decay-power",
        type=float,
        default=1.0,
        help="Bivariate gate decay power (only used with --wht-filter bivariate)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps from config",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = _load_config(args.config)
    if args.steps is not None:
        cfg["training"]["max_steps"] = args.steps

    logger.info(
        "Building ESDM with dataset=%s, wht_filter=%s, a_tau_scale=%s",
        args.dataset, args.wht_filter, args.a_tau_scale,
    )
    model, loader = _build_model(cfg, args)
    logger.info("Model instantiated. dim_input=%d, n_nodes=%d",
                model.dim_input, model.n_nodes)

    # Training loop -- uses the selected data loader when available.
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
                "step=%d  loss=%.4f  tau_n=%.4f  alpha_bar=%.4f  wht=%s  heat_death=%s",
                step, float(out.loss), hc["tau_n"],
                float(model.clock.alpha_bar()),
                args.wht_filter, hc["heat_death"],
            )
        if hc["heat_death"]:
            logger.info("Heat death detected at step %d. Stopping.", step)
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
