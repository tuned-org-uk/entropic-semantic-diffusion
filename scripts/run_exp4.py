"""
Run Experiment 4 — Per-Mode Entropic Schedule on MNIST (PCA embeddings).

Ref: docs/02-EXPERIMENTS.md section Experiment 4

Usage examples
--------------
# Single run with default parameters:
    python scripts/run_exp4.py

# Limited sweep for testing (recommended; ~24 runs):
    python scripts/run_exp4.py --limited-sweep

# Full production sweep:
    python scripts/run_exp4.py --sweep
"""
import argparse
import logging

from esdm.experiments.exp4 import PCA_DIM, run_single, run_sweep


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Experiment 4: per-mode entropic schedule on MNIST-PCA. "
            "Ref: docs/02-EXPERIMENTS.md §Experiment 4."
        )
    )

    # Single-run parameters.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs/exp4")
    parser.add_argument("--dim", type=int, default=PCA_DIM)
    parser.add_argument(
        "--use-per-mode",
        action="store_true",
        help="Use per-mode entropic schedule (default: scalar/global schedule).",
    )
    parser.add_argument(
        "--kl-s-weight",
        type=float,
        default=0.01,
        help="Semantic coupling / barrier height (default: 0.01).",
    )
    parser.add_argument("--lambda-clip", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)

    # Sweep modes.
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the full production sweep.",
    )
    parser.add_argument(
        "--limited-sweep",
        action="store_true",
        help="Run a small test sweep (2 seeds, 3 kl_S_weight, global+per-mode, 200 steps).",
    )

    args = parser.parse_args()

    if args.sweep and args.limited_sweep:
        parser.error("--sweep and --limited-sweep are mutually exclusive.")

    if args.limited_sweep:
        run_sweep(
            dims=[args.dim],
            kl_s_weights=[0.001, 0.01, 0.05],
            seeds=[0, 1],
            use_per_modes=[False, True],
            max_steps=200,
            output_root=args.output_dir,
        )
        print("Experiment 4 limited sweep complete.")
    elif args.sweep:
        run_sweep(
            dims=[args.dim],
            output_root=args.output_dir,
        )
        print("Experiment 4 full sweep complete.")
    else:
        metrics = run_single(
            dim=args.dim,
            use_per_mode=args.use_per_mode,
            kl_S_weight=args.kl_s_weight,
            lambda_clip=args.lambda_clip,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            output_dir=f"{args.output_dir}/seed{args.seed}",
        )
        print(
            f"Experiment 4 single run complete. "
            f"Final test_mse={metrics['test_mse'].dropna().iloc[-1]:.4f} "
            f"knn={metrics['latent_knn_acc'].dropna().iloc[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
