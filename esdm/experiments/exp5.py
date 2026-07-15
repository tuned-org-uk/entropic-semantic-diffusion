r"""
Experiment 5 - Full ESDM training on CIFAR-10 DINO-256 embeddings (issue #22).

Ref: docs/02-EXPERIMENTS.md section Experiment 5
     issue #22 (2026-07-14): full training on real embeddings

This module trains the full ESDM model on the CIFAR-10 DINO-256 loader from
`esdm.data.cifar10`.  It mirrors Experiment 4 and is the high-dimensional
real-data counterpart.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import time
import torch

from ..cli.train import _load_config
from ..data.cifar10 import DEFAULT_OUTPUT_DIM, CIFAR10DINOLoader
from ..graphs.wiring import WireGraph
from ..graphs.wht_filter import GraphWHTFilter
from ..models.esdm import ESDM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DIM: int = DEFAULT_OUTPUT_DIM
N_STEPS: int = 800
A_TAU_SCALE: float = 2.62
GRAD_CLIP: float = 1.0


def _build_l_f(loader, n_samples: int = 5000, k: int = 16, seed: int = 3407) -> torch.Tensor:
    """Build a real feature-space Laplacian L_F from the loader's training data."""
    data = loader.train_data  # (N, F)
    n = data.shape[0]
    if n > n_samples:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=rng)[:n_samples]
        data = data[idx]
    k_eff = min(k, data.shape[1] - 1)
    wg = WireGraph.build(data, k=k_eff, seed=seed)
    return wg.laplacian(normalized=True, sparse=False)


def _build_wht_filter(
    mode: str,
    L_F: torch.Tensor,
    a_tau_scale: float = A_TAU_SCALE,
    tau_decay_power: float = 1.0,
) -> GraphWHTFilter | None:
    """Build a GraphWHTFilter from the real feature-space Laplacian."""
    if mode == "none":
        return None
    from ..cli.wht_filter import build_wht_filter as _factory
    return _factory(mode, L_F.numpy(), a_tau_scale=a_tau_scale, tau_decay_power=tau_decay_power)


def run_single(
    n_steps: int = N_STEPS,
    lr: float = 3e-4,
    wht_filter: str = "none",
    lam_s: float = 1.0,
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
    data_dir: Optional[Path | str] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    pca_fit_samples: int = 500,
) -> pd.DataFrame:
    """
    Run a single Exp-5 training loop on CIFAR-10 DINO-256 embeddings.

    Free parameters: n_steps, lr, wht_filter, lam_s.
    Architecture constants (gamma, n_vibrational_steps, batch_size, grad_clip)
    come from esdm/_configs/default.yaml and AGENTS.md — do not sweep.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    cfg = _load_config("esdm/_configs/default.yaml")
    cfg["model"]["dim_input"] = DIM
    cfg["model"]["n_nodes"] = DIM
    q = max(1, DIM // 4)
    cfg["model"]["dim_latent"] = q
    cfg["model"]["n_modes"] = q
    cfg["training"]["max_steps"] = n_steps
    cfg["model"]["lam_s"] = lam_s
    cfg["clock"]["eps_rel"] = 1e-3  # enable relative heat-death threshold for stochastic batches

    loader = CIFAR10DINOLoader(
        output_dim=DEFAULT_OUTPUT_DIM,
        data_dir=data_dir or "data/cifar10",
        seed=seed,
        pca_fit_samples=pca_fit_samples,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    L_F = _build_l_f(loader, n_samples=min(5000, len(loader.train_data)), seed=seed)
    cfg["model"]["L_base"] = L_F
    cfg["model"]["wht_filter"] = _build_wht_filter(wht_filter, L_F)

    model = ESDM(cfg)

    batch_size = cfg["training"]["batch_size"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Linear warmup over the first 50 steps to avoid destabilising the
    # spectral decoder's variance head.
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    rows: list[dict] = []
    logger.info(
        "Starting Exp5: n_steps=%d batch_size=%d lr=%g wht_filter=%s lam_s=%g "
        "max_train_samples=%s max_test_samples=%s pca_fit_samples=%s device=%s",
        n_steps, batch_size, lr, wht_filter, lam_s,
        max_train_samples, max_test_samples, pca_fit_samples,
        str(next(model.parameters()).device),
    )
    run_start = time.perf_counter()
    for step in range(n_steps):
        x = loader.train_batch(batch_size, seed=seed + step).float()
        optimizer.zero_grad()
        out = model(x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        hc = model.health_check()
        row = {
            "step": step,
            "loss": float(out.loss.detach()),
            "tau_n": hc["tau_n"],
            "alpha_bar": float(model.clock.alpha_bar()),
            "Lambda_tau": hc["Lambda_tau"],
            "heat_death": int(hc["heat_death"]),
            "rho_sign": hc["rho_sign"],
        }
        # Log the ELBO term breakdown from the model output.
        for k, v in out.metrics.items():
            row[k] = float(v.detach())
        if step % cfg["training"].get("log_every", 10) == 0:
            elapsed = time.perf_counter() - run_start
            step_time = elapsed / (step + 1)
            eta = step_time * (n_steps - step - 1)
            logger.info(
                "Exp5 step=%d/%d loss=%.4f tau_n=%.4f alpha_bar=%.4f Lambda_tau=%.4f heat_death=%s "
                "%.2f s/step, ETA %.1f min",
                step, n_steps, row["loss"], row["tau_n"], row["alpha_bar"],
                row["Lambda_tau"], row["heat_death"], step_time, eta / 60.0
            )
        rows.append(row)
        if hc["heat_death"]:
            logger.info("Heat death detected at step %d", step)
            break

    metrics = pd.DataFrame(rows)
    final = metrics.iloc[-1]
    logger.info(
        "Exp5 finished: n_steps=%d final_loss=%.4f tau_n=%.4f alpha_bar=%.4e heat_death=%s",
        len(metrics), final["loss"], final["tau_n"], final["alpha_bar"], final["heat_death"]
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics


def main_cli(
    n_steps: int = N_STEPS,
    wht_filter: str = "none",
    seed: int = 3407,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """CLI entry point for a single Exp-5 run."""
    metrics = run_single(
        n_steps=n_steps,
        wht_filter=wht_filter,
        seed=seed,
        output_dir=output_dir,
    )
    final = metrics.iloc[-1]
    logger.info(
        "Exp5 summary: n_steps=%d, loss=%.4f, tau_n=%.4f, "
        "alpha_bar=%.4e, heat_death=%s",
        len(metrics), final["loss"], final["tau_n"],
        final["alpha_bar"], final["heat_death"]
    )
    return metrics
