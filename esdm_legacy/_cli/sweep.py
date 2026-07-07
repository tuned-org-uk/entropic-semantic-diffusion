"""
esdm-sweep  --  barrier-height sweep: vary kl_S weight and record tau_n curves.
Directly implements Experiment E3 from 01-IMPLEMENTATION-PLAN.md.

Usage:
    uv run esdm-sweep --config esdm/_configs/default.yaml \
        --kl_s_values 0.001 0.01 0.05 0.1 0.5
"""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep kl_S weight")
    parser.add_argument("--config",       default="esdm/_configs/default.yaml")
    parser.add_argument("--kl_s_values",  nargs="+", type=float,
                        default=[0.001, 0.01, 0.05, 0.1, 0.5])
    parser.add_argument("--steps",        type=int, default=2000)
    args = parser.parse_args()

    logger.info("Sweep over kl_S values: %s", args.kl_s_values)
    # Stub: run each kl_S value in sequence when training loop is ready
    logger.info("esdm-sweep: implement after esdm/model.py is complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
