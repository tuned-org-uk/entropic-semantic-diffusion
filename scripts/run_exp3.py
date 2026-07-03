"""
Run Experiment 3 - Barrier Height Sweep on High-Dimensional Gaussian Clusters.

Ref: docs/02-EXPERIMENTS.md section Experiment 3
    issue #6 (updated 2026-07-03): dim=64, bright_modes=dim//2=32

Usage examples
--------------
# Single run with default primary-result parameters:
    python scripts/run_exp3.py

# Single run with custom barrier height and decay:
    python scripts/run_exp3.py --kl-s-weight 0.001 --kl-decay 0.01 --seed 42

# Limited sweep for testing (recommended; ~18 runs, a few minutes):
    python scripts/run_exp3.py --limited-sweep

# Full production sweep (~540 runs, several hours):
    python scripts/run_exp3.py --sweep
"""
import argparse
import logging

from esdm.experiments.exp3 import DIM, BRIGHT_MODES, A_TAU_SCALES, main_cli, run_sweep


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: barrier height sweep on 64-D Gaussian clusters. "
            "dim=64, bright_modes=32 (dim//2). "
            "Ref: issue #6 (updated 2026-07-03)."
        )
    )

    # --- Single-run parameters ---
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/exp3",
        help="Output directory for metrics.csv (default: runs/exp3)",
    )
    parser.add_argument(
        "--kl-decay",
        type=float,
        default=0.03,
        help="KL decay rate (default: 0.03 - primary baseline; 0.08 is Exp-2 control)",
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
        default=10.0,
        help="Entropic pump ceiling (default: 10.0; use 100.0 for Exp-2 control)",
    )
    parser.add_argument(
        "--kl-s-weight",
        type=float,
        default=0.05,
        help="Semantic coupling / barrier height (default: 0.05 - intermediate regime)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=3.0,
        help="Inter-cluster SNR (default: 3.0 - primary result condition)",
    )
    parser.add_argument(
        "--a-tau-scale",
        type=float,
        default=A_TAU_SCALES[-1],
        help=(
            f"Path A gate authority a_tau_scale (default: {A_TAU_SCALES[-1]}; "
            f"sweep uses {A_TAU_SCALES})"
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DIM,
        help=f"Embedding dimension (default: {DIM})",
    )
    parser.add_argument(
        "--bright-modes",
        type=int,
        default=BRIGHT_MODES,
        help=(
            f"Number of bright/active spectral modes (default: {BRIGHT_MODES} = dim//2). "
            "WHT gate dimension equals bright_modes."
        ),
    )

    # --- Sweep mode ---
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run the full production sweep (~540 runs, several hours). "
            "For testing use --limited-sweep instead. "
            "Ignores --seed, --kl-s-weight, --kl-decay, --rho, --lambda-clip, --a-tau-scale."
        ),
    )
    parser.add_argument(
        "--limited-sweep",
        action="store_true",
        help=(
            "Run a small test sweep (~18 runs, a few minutes). "
            "Recommended for CI/development; ignores single-run parameters."
        ),
    )
    parser.add_argument(
        "--assert-each",
        action="store_true",
        help="Assert success criteria after every individual sweep run (slow).",
    )

    args = parser.parse_args()

    if args.sweep and args.limited_sweep:
        parser.error("--sweep and --limited-sweep are mutually exclusive.")

    if args.sweep:
        run_sweep(
            dim=args.dim,
            bright_modes=args.bright_modes,
            output_root=args.output_dir,
            assert_each=args.assert_each,
        )
        print("Experiment 3 full sweep complete.")
    elif args.limited_sweep:
        run_sweep(
            dim=args.dim,
            bright_modes=args.bright_modes,
            kl_s_weights=[0.001, 0.05, 1.0],
            kl_decays=[0.03, 0.08],
            rho_snrs=[3.0],
            lambda_clips=[10.0],
            a_tau_scales=A_TAU_SCALES,
            seeds=[0, 1],
            n_steps=200,
            output_root=args.output_dir,
            assert_each=args.assert_each,
        )
        print("Experiment 3 limited sweep complete.")
    else:
        main_cli(
            output_dir=f"{args.output_dir}/{args.seed}",
            seed=args.seed,
            kl_decay=args.kl_decay,
            clock_eps=args.clock_eps,
            lambda_clip=args.lambda_clip,
            kl_S_weight=args.kl_s_weight,
            rho=args.rho,
            a_tau_scale=args.a_tau_scale,
            dim=args.dim,
            bright_modes=args.bright_modes,
        )
        print("Experiment 3 single run complete - all assertions passed.")


if __name__ == "__main__":
    main()
