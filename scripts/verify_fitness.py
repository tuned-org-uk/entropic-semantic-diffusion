#!/usr/bin/env python3
"""Verify command: output ESDM fitness score as a single number."""
import argparse

from esdm.eval.fitness import compute_fitness, compute_fitness_real
from esdm.experiments import exp3, exp4, exp5


def _dataset_choices() -> list[str]:
    return ["exp3", "mnist", "cifar10"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify ESDM fitness for a given dataset/experiment."
    )
    parser.add_argument(
        "--dataset",
        choices=_dataset_choices(),
        default="exp3",
        help='Dataset/experiment to evaluate (default: exp3).',
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=800,
        help='Number of training steps (default: 800).',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help='Learning rate for real-data runs (default: 3e-4).',
    )
    parser.add_argument(
        "--wht-filter",
        type=str,
        default="none",
        help='WHT filter mode: none, basic, bivariate (default: none).',
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help='CIFAR-10 train subset size for fast verification (default: 2000).',
    )
    parser.add_argument(
        "--pca-fit-samples",
        type=int,
        default=1000,
        help='CIFAR-10 PCA-fit subset size (default: 1000).',
    )
    parser.add_argument(
        "--lam-s",
        type=float,
        default=1.0,
        help='Spectral prior precision multiplier (default: 1.0).',
    )
    args = parser.parse_args(argv)

    if args.dataset == "exp3":
        m = exp3.run_single(
            n_steps=args.n_steps,
            seed=3407,
            kl_S_weight=0.5,
            kl_decay=0.03,
            wht_filter="basic",
            a_tau_scale=2.62,
        )
        r = compute_fitness(m)
    elif args.dataset == "mnist":
        m = exp4.run_single(
            n_steps=args.n_steps,
            lr=args.lr,
            wht_filter=args.wht_filter,
            seed=3407,
        )
        r = compute_fitness_real(m)
    elif args.dataset == "cifar10":
        m = exp5.run_single(
            n_steps=args.n_steps,
            lr=args.lr,
            wht_filter=args.wht_filter,
            seed=3407,
            max_train_samples=args.max_train_samples,
            max_test_samples=200,
            pca_fit_samples=args.pca_fit_samples,
            lam_s=args.lam_s,
        )
        r = compute_fitness_real(m)
    else:
        parser.error(f"Unhandled dataset: {args.dataset}")
        return 1

    print(f'{r["total"]:.2f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
