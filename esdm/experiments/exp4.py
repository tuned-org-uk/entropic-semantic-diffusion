r"""
Experiment 4 - Full ESDM training on MNIST PCA-16 embeddings (issue #22).

Ref: docs/02-EXPERIMENTS.md section Experiment 4
     issue #22 (2026-07-14): full training on real embeddings

This module trains the full ESDM model on the MNIST PCA-16 loader from
`esdm.data.mnist`.  It reuses the training logic from `esdm.cli.train` and
logs the same health diagnostics used in Experiment 3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..cli.train import _build_model, _load_config
from ..data.mnist import DEFAULT_PCA_DIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DIM: int = DEFAULT_PCA_DIM
N_STEPS: int = 200
BATCH_SIZE: int = 64
A_TAU_SCALE: float = 2.62


class _TrainArgs:
    """Minimal namespace for `_build_model` (mirrors train CLI args)."""

    def __init__(
        self,
        dataset: str = "mnist",
        wht_filter: str = "basic",
        a_tau_scale: float = A_TAU_SCALE,
        tau_decay_power: float = 1.0,
        seed: int = 3407,
    ) -> None:
        self.dataset = dataset
        self.wht_filter = wht_filter
        self.a_tau_scale = a_tau_scale
        self.tau_decay_power = tau_decay_power
        self.seed = seed


def run_single(
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    a_tau_scale: float = A_TAU_SCALE,
    tau_decay_power: float = 1.0,
    wht_filter: str = "basic",
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
    data_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Run a single Exp-4 training loop on MNIST PCA-16 embeddings.

    Parameters
    ----------
    wht_filter : str
        "none" / "basic" / "bivariate" -- passed to the WHT filter factory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = DIM
    cfg["model"]["n_nodes"] = DIM
    cfg["model"]["dim_latent"] = min(cfg["model"].get("dim_latent", DIM), DIM // 2)
    cfg["model"]["n_modes"] = min(cfg["model"].get("n_modes", DIM), DIM // 2)
    cfg["training"]["max_steps"] = n_steps
    cfg["training"]["batch_size"] = batch_size

    args = _TrainArgs(
        dataset="mnist",
        wht_filter=wht_filter,
        a_tau_scale=a_tau_scale,
        tau_decay_power=tau_decay_power,
        seed=seed,
    )
    model, loader = _build_model(cfg, args)
    model = model.double()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    rows: list[dict] = []
    for step in range(n_steps):
        x = loader.train_batch(batch_size, seed=seed + step)
        optimizer.zero_grad()
        out = model(x)
        out.loss.backward()
        optimizer.step()

        hc = model.health_check()
        row = {
            "step": step,
            "loss": float(out.loss.detach()),
            "tau_n": hc["tau_n"],
            "alpha_bar": float(model.clock.alpha_bar()),
            "Lambda_tau": hc["Lambda_tau"],
            "heat_death": int(hc["heat_death"]),
            "conserved": hc["conserved"],
            "rho_sign": hc["rho_sign"],
        }
        if step % cfg["training"].get("log_every", 10) == 0:
            logger.info(
                "Exp4 step=%d loss=%.4f tau_n=%.4f alpha_bar=%.4f heat_death=%s",
                step, row["loss"], row["tau_n"], row["alpha_bar"], row["heat_death"]
            )
        rows.append(row)
        if hc["heat_death"]:
            logger.info("Heat death detected at step %d", step)
            break

    metrics = pd.DataFrame(rows)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics


def main_cli(
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    a_tau_scale: float = A_TAU_SCALE,
    wht_filter: str = "basic",
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """CLI entry point for a single Exp-4 run."""
    metrics = run_single(
        n_steps=n_steps,
        batch_size=batch_size,
        a_tau_scale=a_tau_scale,
        wht_filter=wht_filter,
        seed=seed,
        output_dir=output_dir,
    )
    final = metrics.iloc[-1]
    logger.info(
        "Exp4 summary: n_steps=%d, loss=%.4f, tau_n=%.4f, "
        "alpha_bar=%.4e, heat_death=%s",
        len(metrics), final["loss"], final["tau_n"],
        final["alpha_bar"], final["heat_death"]
    )
    return metrics
