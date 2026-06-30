"""
Run Experiment 1 — Entropic Clock Calibration on a Two-Mode Synthetic Graph.

Ref: docs/02-EXPERIMENTS.md §Experiment 1
"""
import argparse
import logging

from esdm.experiments.exp1 import run_bridge_sweep


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Experiment 1: entropic clock calibration"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/exp1",
        help="Base output directory (default: runs/exp1)",
    )
    args = parser.parse_args()

    run_bridge_sweep(
        bridge_weights=(0.0, 0.01, 0.1, 1.0),
        base_output_dir=f"{args.output_dir}/{args.seed}",
        seed=args.seed,
    )
    print("Experiment 1 complete — all assertions passed.")


if __name__ == "__main__":
    main()
