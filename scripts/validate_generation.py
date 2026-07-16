#!/usr/bin/env python3
"""
Issue #29 Phase 1 - Validate generation on real data.

Trains ESDM on CIFAR-10 DINO-256, saves checkpoint, runs reverse-SDE
sampler, and validates generated embeddings against the real test
distribution via statistics, k-NN, and nearest-neighbor distances.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from esdm.experiments import exp5
from esdm.sampling.entropic_sampler import entropic_reverse_sde
from esdm.sampling.q_space_sampler import q_space_reverse_sde

logger = logging.getLogger("esdm.validate_generation")


def _embedding_stats(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "norm_mean": float(x.norm(dim=-1).mean()),
        "norm_std": float(x.norm(dim=-1).std()),
        "dim": x.shape[1],
        "n": x.shape[0],
    }


def _nearest_neighbor_distances(gen, real):
    gen_f = gen.float()
    real_f = real.float()
    dists = torch.cdist(gen_f, real_f)
    nn_dists = dists.min(dim=-1).values
    return {
        "nn_dist_mean": float(nn_dists.mean()),
        "nn_dist_std": float(nn_dists.std()),
        "nn_dist_min": float(nn_dists.min()),
        "nn_dist_max": float(nn_dists.max()),
        "nn_dist_median": float(nn_dists.median()),
    }


def _real_nn_distances(real_test, n_samples=64, seed=3407):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(real_test.shape[0], generator=rng)[:n_samples + 1]
    subset = real_test[idx].float()
    dists = torch.cdist(subset[:n_samples], subset[1:])
    nn_dists = dists.min(dim=-1).values
    return {
        "nn_dist_mean": float(nn_dists.mean()),
        "nn_dist_std": float(nn_dists.std()),
        "nn_dist_min": float(nn_dists.min()),
        "nn_dist_max": float(nn_dists.max()),
        "nn_dist_median": float(nn_dists.median()),
    }


def _knn_diversity(gen, real_train, real_labels, k=5):
    gen_f = gen.float()
    real_f = real_train.float()
    gen_n = gen_f / gen_f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    real_n = real_f / real_f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sims = gen_n @ real_n.T
    _, idx = sims.topk(k, dim=-1)
    knn_labels = real_labels[idx.cpu().numpy()]
    from scipy.stats import mode
    preds = mode(knn_labels, axis=1, keepdims=False).mode
    return int(len(np.unique(preds)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="validate-generation",
        description="Issue #29 Phase 1: validate ESDM generation on real CIFAR-10.",
    )
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--sampler-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--pca-fit-samples", type=int, default=1000)
    parser.add_argument("--sampler", choices=["ddpm", "qspace"], default="qspace",
                        help="Sampler to use (default: qspace)")
    parser.add_argument("--output-dir", default="results/phase1_generation/")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train ESDM and keep the model
    logger.info("=== Step 1: Training ESDM on CIFAR-10 ===")
    t0 = time.perf_counter()
    metrics, model, loader = exp5.run_single(
        n_steps=args.n_steps,
        lr=args.lr,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        pca_fit_samples=args.pca_fit_samples,
        return_model=True,
    )
    train_time = time.perf_counter() - t0
    logger.info("Training complete in %.1fs (%d steps)", train_time, len(metrics))

    # Step 2: Save checkpoint
    ckpt_path = out_dir / "esdm_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "dim_input": model.dim_input,
            "n_nodes": model.n_nodes,
            "dim_latent": model.dim_latent,
            "n_modes": model.n_modes,
        },
        "training": {
            "n_steps": len(metrics),
            "lr": args.lr,
            "seed": args.seed,
            "final_loss": float(metrics.iloc[-1]["loss"]),
            "final_tau_n": float(metrics.iloc[-1]["tau_n"]),
            "final_alpha_bar": float(metrics.iloc[-1]["alpha_bar"]),
            "heat_death": int(metrics.iloc[-1]["heat_death"]),
        },
    }, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)

    # Step 3: Run reverse-SDE sampler
    sampler_name = args.sampler
    logger.info("=== Step 2: Running %s sampler (%d samples, %d steps) ===",
                sampler_name, args.n_samples, args.sampler_steps)
    t0 = time.perf_counter()
    if sampler_name == "qspace":
        generated = q_space_reverse_sde(
            model, n_samples=args.n_samples, n_steps=args.sampler_steps, seed=args.seed,
        )
    else:
        generated = entropic_reverse_sde(
            model, n_samples=args.n_samples, n_steps=args.sampler_steps, seed=args.seed,
        )
    sample_time = time.perf_counter() - t0
    logger.info("Sampling complete in %.1fs, output shape: %s", sample_time, tuple(generated.shape))
    torch.save(generated, out_dir / "generated_embeddings.pt")
    logger.info("Generated embeddings saved to %s", out_dir / "generated_embeddings.pt")

    # Step 4: Validate against real test distribution
    logger.info("=== Step 3: Validating generated embeddings ===")
    real_test = loader.test_data
    real_train = loader.train_data

    gen_stats = _embedding_stats(generated)
    real_stats = _embedding_stats(real_test)
    logger.info("Generated stats: mean=%.4f std=%.4f norm=%.4f+/-%.4f",
                gen_stats["mean"], gen_stats["std"], gen_stats["norm_mean"], gen_stats["norm_std"])
    logger.info("Real test stats:  mean=%.4f std=%.4f norm=%.4f+/-%.4f",
                real_stats["mean"], real_stats["std"], real_stats["norm_mean"], real_stats["norm_std"])

    gen_nn = _nearest_neighbor_distances(generated, real_test)
    real_nn = _real_nn_distances(real_test, n_samples=args.n_samples, seed=args.seed)
    logger.info("Gen->Real NN dist: mean=%.4f median=%.4f min=%.4f max=%.4f",
                gen_nn["nn_dist_mean"], gen_nn["nn_dist_median"], gen_nn["nn_dist_min"], gen_nn["nn_dist_max"])
    logger.info("Real->Real NN dist: mean=%.4f median=%.4f min=%.4f max=%.4f",
                real_nn["nn_dist_mean"], real_nn["nn_dist_median"], real_nn["nn_dist_min"], real_nn["nn_dist_max"])

    ratio = gen_nn["nn_dist_mean"] / max(real_nn["nn_dist_mean"], 1e-12)
    logger.info("NN distance ratio (gen/real): %.4f (1.0=identical, >1=off-manifold)", ratio)

    try:
        train_labels = loader._train_labels
        n_unique = _knn_diversity(generated, real_train, train_labels, k=5)
        logger.info("k-NN diversity: %d unique classes among %d generated samples", n_unique, args.n_samples)
    except Exception as e:
        logger.warning("k-NN diversity check failed: %s", e)
        n_unique = -1

    # Step 5: Write validation report
    report = {
        "training_time_s": train_time,
        "sampling_time_s": sample_time,
        "n_steps_trained": len(metrics),
        "n_samples_generated": args.n_samples,
        "sampler": sampler_name,
        "sampler_steps": args.sampler_steps,
        "generated_stats": gen_stats,
        "real_test_stats": real_stats,
        "generated_nn_distances": gen_nn,
        "real_nn_distances": real_nn,
        "nn_distance_ratio": ratio,
        "knn_unique_classes": n_unique,
        "verdict": "PASS" if ratio < 2.0 else "FAIL",
    }

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Validation report saved to %s", report_path)
    logger.info("=== Phase 1 Verdict: %s (NN ratio=%.4f) ===", report["verdict"], ratio)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
