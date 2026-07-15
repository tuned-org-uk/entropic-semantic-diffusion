"""
ESDM barrier-height sweep CLI.

Sweeps kl_S weight (the semantic coupling strength / barrier height V
analogue from Barontini 2026) and logs tau_n vs. step curves.

Usage:
    esdm-sweep --wht-filter bivariate --kl-s-values 0.001,0.01,0.1,1.0
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import yaml

from esdm.cli.wht_filter import build_wht_filter, WHT_FILTER_CHOICES
from esdm.models.esdm import ESDM

logger = logging.getLogger("esdm.cli.sweep")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-sweep",
        description="Sweep kl_S weight (barrier height V) and WHT filter modes.",
    )
    parser.add_argument("--config", "-c", default="esdm/_configs/default.yaml")
    parser.add_argument(
        "--kl-s-values",
        default="0.01,0.1,1.0",
        help="Comma-separated kl_S weights to sweep",
    )
    parser.add_argument(
        "--wht-filter",
        choices=WHT_FILTER_CHOICES,
        default="none",
        help="WHT spectral filter directive (default: none)",
    )
    parser.add_argument("--a-tau-scale", type=float, default=5.0)
    parser.add_argument("--tau-decay-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(argv)

    kl_values = [float(v) for v in args.kl_s_values.split(",")]
    logger.info("Sweep: kl_S=%s, wht_filter=%s", kl_values, args.wht_filter)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    for kl_s in kl_values:
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_cfg.items()}
        cfg.setdefault("elbo", {})["kl_s_weight"] = kl_s
        cfg["training"]["max_steps"] = args.steps

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.wht_filter != "none":
            F = cfg["model"]["dim_input"]
            rng = np.random.default_rng(args.seed)
            A = rng.standard_normal((F, F))
            L_F = A.T @ A
            np.fill_diagonal(L_F, L_F.diagonal() + 1e-3)
            L_F = (L_F + L_F.T) / 2.0
            cfg["model"]["wht_filter"] = build_wht_filter(
                args.wht_filter, L_F,
                a_tau_scale=args.a_tau_scale,
                tau_decay_power=args.tau_decay_power,
            )
        else:
            cfg["model"]["wht_filter"] = None

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
            "kl_S=%.4f  wht=%s  final_tau=%.4f  steps_run=%d  alpha_bar=%.4f",
            kl_s, args.wht_filter, taus[-1] if taus else 0.0,
            len(taus), float(model.clock.alpha_bar()),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
