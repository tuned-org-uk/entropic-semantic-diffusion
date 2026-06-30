"""
Run Experiment 2 — Vibrational Wave Recurrence on a 1-D Harmonic Chain.

Ref: docs/02-EXPERIMENTS.md §Experiment 2
"""
import argparse
import logging

from esdm.experiments.exp2 import main_cli


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Experiment 2: vibrational wave recurrence on a path graph"
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
        default="runs/exp2",
        help="Output directory (default: runs/exp2)",
    )
    args = parser.parse_args()

    main_cli(
        output_dir=f"{args.output_dir}/{args.seed}",
        seed=args.seed,
    )
    print("Experiment 2 complete — all assertions passed.")


if __name__ == "__main__":
    main()
