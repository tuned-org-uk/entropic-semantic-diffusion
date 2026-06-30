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
    parser.add_argument(
        "--kl-decay",
        type=float,
        default=0.08,
        help="KL decay rate (default: 0.08)",
    )
    parser.add_argument(
        "--clock-eps",
        type=float,
        default=1e-8,
        help="Clock flatness threshold (default: 1e-8)",
    )
    parser.add_argument(
        "--lambda-clip",
        type=float,
        default=100.0,
        help="Entropic pump ceiling (default: 100.0)",
    )
    parser.add_argument(
        "--bright-modes",
        type=int,
        default=16,
        help="Number of bright/active modes for dark-entropy proxy (default: 16)",
    )
    args = parser.parse_args()

    main_cli(
        output_dir=f"{args.output_dir}/{args.seed}",
        seed=args.seed,
        kl_decay=args.kl_decay,
        clock_eps=args.clock_eps,
        lambda_clip=args.lambda_clip,
        bright_modes=args.bright_modes,
    )
    print("Experiment 2 complete — all assertions passed.")


if __name__ == "__main__":
    main()
