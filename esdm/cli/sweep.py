"""
ESDM barrier-height sweep CLI.

Sweeps kl_S weight (the semantic coupling strength / barrier height V
analogue from Barontini 2026) and logs tau_n vs. step curves.

Usage:
    esdm-sweep --kl-s-values 0.001,0.01,0.1,1.0
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import yaml

from esdm.models.esdm import ESDM

logger = logging.getLogger("esdm.cli.sweep")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-sweep",
        description="Sweep kl_S weight (barrier height V).",
    )
    parser.add_argument("--config", "-c", default="esdm/_configs/default.yaml")
    parser.add_argument(
        "--kl-s-values",
        default="0.01,0.1,1.0",
        help="Comma-separated kl_S weights to sweep",
    )
    parser.add_argument("--a-tau-scale", type=float, default=5.0)
    parser.add_argument("--tau-decay-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(argv)

    kl_values = [float(v) for v in args.kl_s_values.split(",")]
    logger.info("Sweep: kl_S=%s", kl_values)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    for kl_s in kl_values:
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_cfg.items()}
        cfg.setdefault("elbo", {})["kl_s_weight"] = kl_s
        cfg["training"]["max_steps"] = args.steps

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        
        model = ESDM(cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        batch_size = cfg["training"]["batch_size"]

        taus = []
        for step in range(args.steps):
            x = torch.randn(batch_size, model.dim_input)
            optimizer.zero_grad()
            out = model(x)
            out.loss.backward()
            optimizer.step()
            taus.append(out.clock.tau_n)
            if model.health_check()["heat_death"]:
                break

        logger.info(
            "kl_S=%.4f  final_tau=%.4f  steps_run=%d  alpha_bar=%.4f",
            kl_s, taus[-1] if taus else 0.0,
            len(taus), float(model.clock.alpha_bar()),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
