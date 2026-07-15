#!/usr/bin/env python3
"""Verify command: output ESDM fitness score as a single number."""
import argparse

from esdm.eval.fitness import compute_fitness_real
from esdm.experiments import exp4, exp5


def _dataset_choices() -> list[str]:
    return ["mnist", "cifar10"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify ESDM fitness for a given dataset/experiment."
    )
    parser.add_argument(
        "--dataset",
        choices=_dataset_choices(),
        default="cifar10",
        help='Dataset/experiment to evaluate (default: cifar10).',
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
    parser.add_argument(
        "--q",
        type=int,
        default=None,
        help='Spectral modes q for hybridisation operator (default: F/4).',
    )
    parser.add_argument(
        "--sigma-sq",
        type=float,
        default=None,
        help='Diffusion coefficient sigma^2 for H operator (default: 1.0).',
    )
    parser.add_argument(
        "--freeze-h",
        action="store_true",
        default=False,
        help='Freeze H operator params (sigma^2, eta) for ablation.',
    )
    args = parser.parse_args(argv)

    kwargs = {
        "n_steps": args.n_steps,
        "lr": args.lr,
        "lam_s": args.lam_s,
        "seed": 3407,
        "q": args.q,
        "sigma_sq": args.sigma_sq,
        "freeze_h": args.freeze_h,
    }

    if args.dataset == "mnist":
        m = exp4.run_single(**kwargs)
    elif args.dataset == "cifar10":
        m = exp5.run_single(
            **kwargs,
            max_train_samples=args.max_train_samples,
            max_test_samples=200,
            pca_fit_samples=args.pca_fit_samples,
        )
    else:
        parser.error(f"Unhandled dataset: {args.dataset}")
        return 1

    r = compute_fitness_real(m)
    print(f'{r["total"]:.2f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
