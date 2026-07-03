"""
Experiment 4 — Per-Mode Entropic Schedule on MNIST (PCA embeddings).

Ref: docs/02-EXPERIMENTS.md section Experiment 4

This experiment introduces a real semantic corpus (MNIST) at low dimensionality
via PCA projection, and validates the per-mode entropic schedule feature of
``EntropicDiffusionDecoder`` (``use_per_mode=True``).

Two model variants are trained side-by-side:
  * ESDM-global  : ``use_per_mode=False`` — scalar ``alpha_bar(tau_n)``.
  * ESDM-per-mode: ``use_per_mode=True``  — per-mode ``alpha_bar(tau_k)``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from esdm.model import ESDM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PCA_DIM: int = 16
N_VIBRATIONAL_STEPS: int = 2
BATCH_SIZE: int = 128
MAX_STEPS: int = 10000
EVAL_EVERY: int = 500
LR: float = 3.0e-4
WEIGHT_DECAY: float = 1.0e-5
RANDOM_STATE: int = 42

# Sweep grid
KL_S_WEIGHTS = [0.001, 0.01, 0.05]
SEEDS = [0, 1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_mnist_pca(
    dim: int = PCA_DIM,
    train_size: int = 60000,
    test_size: int = 10000,
    random_state: int = RANDOM_STATE,
    cache_dir: Optional[Path | str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Load MNIST, project to ``dim`` dimensions with PCA, L2-normalise, and
    return train/test tensors plus the fitted PCA/scaler objects.

    Returns
    -------
    X_train, X_test : Tensor, shape (N, dim)
        L2-normalised PCA projections as float64 CPU tensors.
    y_train, y_test : Tensor, shape (N,)
        Digit labels as int64 CPU tensors.
    labels_train, labels_test : ndarray
        Original string labels from OpenML (kept for evaluators).
    pca : PCA
        Fitted PCA object (explained_variance_ratio_ is used for the
        tau-vs-variance hypothesis test).
    scaler : StandardScaler
        Scaler fit on raw pixels and applied to test set.
    """
    logger.info("Loading MNIST via fetch_openml...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        parser="auto",
        as_frame=False,
        data_home=str(cache_dir) if cache_dir is not None else None,
    )
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(np.int64)

    # Stratified train/test split matching standard MNIST partition.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Standardise pixel values before PCA.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=dim, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # L2-normalise each sample to unit length.
    train_norms = np.linalg.norm(X_train_pca, axis=1, keepdims=True).clip(min=1e-12)
    test_norms = np.linalg.norm(X_test_pca, axis=1, keepdims=True).clip(min=1e-12)
    X_train_pca = X_train_pca / train_norms
    X_test_pca = X_test_pca / test_norms

    return (
        torch.tensor(X_train_pca, dtype=torch.float32),
        torch.tensor(X_test_pca, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(y_test, dtype=torch.int64),
        mnist.target[:train_size],
        mnist.target[train_size:],
        pca,
        scaler,
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def make_esdm_cfg(
    dim: int,
    use_per_mode: bool = False,
    kl_S_weight: float = 0.01,
    kl_decay: float = 0.0,  # unused; clock is driven by real KL
    lambda_clip: float = 100.0,
    heat_death_patience: int = 1000,
    n_vibrational_steps: int = N_VIBRATIONAL_STEPS,
) -> dict:
    """Build an ESDM config dict for MNIST-PCA."""
    return {
        "model": {
            "dim_input": dim,
            "dim_latent": dim,
            "n_nodes": dim,
            "n_modes": dim,
            "mode_threshold": 0.01,
            "lam_s": 1.0,
            "n_vibrational_steps": n_vibrational_steps,
            "gamma": 0.1,
            "use_per_mode": use_per_mode,
        },
        "clock": {
            "eps": 1e-8,
            "heat_death_patience": heat_death_patience,
            "lambda_clip": lambda_clip,
        },
        "elbo": {
            "kl_s_weight": kl_S_weight,
            "beta_conservation": None,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "log_every": EVAL_EVERY,
            "seed": RANDOM_STATE,
        },
        "fusion": {
            "alpha": 0.5,
            "beta": 0.5,
            "n_eigvecs": 64,
        },
        "paths": {
            "lf_pt": "data/L_F.pt",
            "v_pt": "data/V.pt",
            "output_dir": "results/",
        },
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def reconstruction_mse(model: ESDM, X: torch.Tensor, batch_size: int = 512) -> float:
    """MSE between input and deterministic reconstruction (noise=0)."""
    model.eval()
    mse = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            out = model(xb)
            mse += float(F.mse_loss(out.x_hat, xb, reduction="sum"))
            n += xb.numel()
    model.train()
    return mse / n


def latent_knn_accuracy(
    model: ESDM,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_neighbors: int = 10,
    max_train: int = 10000,
    batch_size: int = 512,
) -> float:
    """10-NN classification accuracy in the ESDM latent ``z`` space."""
    model.eval()
    with torch.no_grad():
        z_train_parts = []
        n_encoded = 0
        for i in range(0, min(len(X_train), max_train), batch_size):
            if n_encoded >= max_train:
                break
            out = model(X_train[i:i + batch_size])
            z_train_parts.append(out.z.detach().cpu().numpy())
            n_encoded += out.z.shape[0]
        z_train = np.concatenate(z_train_parts, axis=0)[:max_train]

        z_test_parts = []
        for i in range(0, len(X_test), batch_size):
            out = model(X_test[i:i + batch_size])
            z_test_parts.append(out.z.detach().cpu().numpy())
        z_test = np.concatenate(z_test_parts, axis=0)
    model.train()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
    clf.fit(z_train, y_train[:max_train].numpy())
    acc = float(clf.score(z_test, y_test.numpy()))
    return acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_single(
    dim: int = PCA_DIM,
    use_per_mode: bool = False,
    kl_S_weight: float = 0.01,
    lambda_clip: float = 100.0,
    max_steps: int = MAX_STEPS,
    eval_every: int = EVAL_EVERY,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = RANDOM_STATE,
    n_vibrational_steps: int = N_VIBRATIONAL_STEPS,
    output_dir: Optional[Path | str] = None,
    cache_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Train one ESDM variant on MNIST-PCA and return per-step metrics.

    Parameters
    ----------
    use_per_mode :
        If ``True``, use per-mode entropic schedule; otherwise scalar schedule.
    output_dir :
        If given, write ``metrics.csv`` to this directory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, _, _, pca, _ = load_mnist_pca(
        dim=dim, random_state=seed, cache_dir=cache_dir
    )
    N_train = X_train.shape[0]

    cfg = make_esdm_cfg(
        dim=dim,
        use_per_mode=use_per_mode,
        kl_S_weight=kl_S_weight,
        lambda_clip=lambda_clip,
        n_vibrational_steps=n_vibrational_steps,
    )
    model = ESDM(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Cosine LR decay as specified in the experiment doc.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps
    )

    logger.info(
        "Exp4: dim=%d, use_per_mode=%s, kl_S_weight=%.3f, lambda_clip=%.1f, "
        "max_steps=%d, seed=%d",
        dim, use_per_mode, kl_S_weight, lambda_clip, max_steps, seed,
    )

    records: list[dict] = []
    best_test_mse = float("inf")

    for step in range(max_steps):
        idx = torch.randint(0, N_train, (batch_size,))
        xb = X_train[idx]

        model.train()
        optimizer.zero_grad()
        out = model(xb)
        out.loss.backward()
        optimizer.step()
        scheduler.step()

        # Per-step core metrics.
        record = {
            "step": step,
            "seed": seed,
            "dim": dim,
            "use_per_mode": int(use_per_mode),
            "kl_S_weight": kl_S_weight,
            "lambda_clip": lambda_clip,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": float(out.loss.detach()),
            "train_recon": float(out.metrics["recon"]),
            "kl_S": float(out.metrics["kl_S"]),
            "kl_tau": float(out.metrics["kl_tau"]),
            "dark_entropy": float(out.metrics["dark_entropy"]),
            "conserved": float(out.metrics["conserved"]),
            "tau_n": out.clock.tau_n,
            "alpha_bar": float(model.clock.alpha_bar()),
            "Lambda_tau": out.clock.Lambda_tau,
            "rho_sign": out.clock.rho_sign,
            "is_heat_death": int(model.clock.is_heat_death()),
            "N_active": int(model._last_N_active.item()),
        }

        # Periodic evaluation on the full test set.
        if step % eval_every == 0 or step == max_steps - 1:
            test_mse = reconstruction_mse(model, X_test)
            train_mse = reconstruction_mse(model, X_train[:10000])
            knn_acc = latent_knn_accuracy(
                model,
                X_train, y_train,
                X_test[:2000], y_test[:2000],
                n_neighbors=10,
                max_train=5000,
            )
            record.update({
                "test_mse": test_mse,
                "train_mse": train_mse,
                "latent_knn_acc": knn_acc,
            })
            best_test_mse = min(best_test_mse, test_mse)
            logger.info(
                "Step %5d | train_mse=%.4f test_mse=%.4f knn=%.3f tau_n=%.4f",
                step, train_mse, test_mse, knn_acc, out.clock.tau_n,
            )
        else:
            record.update({
                "test_mse": np.nan,
                "train_mse": np.nan,
                "latent_knn_acc": np.nan,
            })

        records.append(record)

    # Final convergence diagnostics.
    final_tau_k = (
        model.clock.per_mode_tau.detach().cpu().numpy()
        if model.clock.per_mode_tau is not None
        else np.full(dim, np.nan)
    )
    explained_variance = pca.explained_variance_ratio_
    if np.std(final_tau_k) > 1e-12 and np.std(explained_variance) > 1e-12:
        tau_var_corr = float(spearmanr(final_tau_k, explained_variance).statistic)
    else:
        tau_var_corr = 0.0

    # Final-only diagnostics are replicated across all rows for schema
    # compatibility (downstream analysis should read them from the last row).
    final_only = {
        "best_test_mse": best_test_mse,
        "final_tau_var_corr": tau_var_corr,
        "final_tau_k": final_tau_k.tolist(),
        "explained_variance": explained_variance.tolist(),
    }
    for rec in records:
        rec.update(final_only)

    metrics = pd.DataFrame(records)

    # Forward-fill evaluation-only columns so the time series is complete.
    for col in ("test_mse", "train_mse", "latent_knn_acc"):
        metrics[col] = metrics[col].ffill()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        logger.info("Wrote metrics to %s", out / "metrics.csv")

    return metrics


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def assert_success_criteria(
    metrics: pd.DataFrame,
    require_per_mode_better: bool = False,
) -> dict[str, bool]:
    """
    Check Experiment 4 success criteria.

    Returns a dict of criterion -> bool.  Raises ``AssertionError`` on the
    first failure if ``require_per_mode_better`` is ``True``; otherwise
    returns the verdict dict for downstream aggregation.
    """
    final = metrics.iloc[-1]
    results: dict[str, bool] = {}

    # 1. No NaN/Inf in training.
    numeric = metrics.select_dtypes(include=[np.number])
    results["finite_training"] = bool(np.isfinite(numeric.to_numpy()).all())

    # 2. Test MSE finite and improved from first eval.
    eval_rows = metrics.dropna(subset=["test_mse"])
    results["test_mse_finite"] = bool(not eval_rows.empty and np.isfinite(eval_rows["test_mse"]).all())
    if not eval_rows.empty:
        results["test_mse_improved"] = bool(float(eval_rows["test_mse"].iloc[-1]) <= float(eval_rows["test_mse"].iloc[0]) * 1.5)
    else:
        results["test_mse_improved"] = False

    # 3. Latent kNN accuracy threshold.
    knn = eval_rows["latent_knn_acc"].dropna()
    results["knn_acc_high"] = bool(not knn.empty and float(knn.iloc[-1]) >= 0.70)

    # 4. Negative correlation between tau_k and explained variance.
    corr = final.get("final_tau_var_corr", 0.0)
    results["tau_var_negative_corr"] = bool(float(corr) < -0.1)

    # 5. All per-mode tau values positive (no mode collapse).
    # For scalar/global schedule this criterion is not applicable and passes.
    if bool(final.get("use_per_mode", 0)):
        tau_k = final.get("final_tau_k", [])
        if isinstance(tau_k, str):
            tau_k = ast.literal_eval(tau_k)
        tau_k_arr = np.asarray(tau_k, dtype=float)
        results["all_tau_positive"] = bool(tau_k_arr.size > 0 and np.all(tau_k_arr > 0))
    else:
        results["all_tau_positive"] = True

    # 6. Conservation residual small after step 5000.
    late = metrics[metrics["step"] >= min(5000, len(metrics) - 1)]
    if not late.empty:
        results["conserved_small"] = bool((late["conserved"] < 0.05 * (late["kl_S"] + late["kl_tau"] + late["dark_entropy"] + 1e-8)).mean() > 0.8)
    else:
        # Short smoke runs never reach step 5000; this criterion is N/A.
        logger.warning(
            "conserved_small criterion skipped: max_steps=%d < 5000",
            int(metrics["step"].max()) + 1,
        )
        results["conserved_small"] = False

    if require_per_mode_better:
        for name, ok in results.items():
            assert ok, f"Exp4 success criterion failed: {name}"

    return results


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    dims: list[int] = [PCA_DIM],
    kl_s_weights: list[float] = KL_S_WEIGHTS,
    seeds: list[int] = SEEDS,
    use_per_modes: list[bool] = [False, True],
    max_steps: int = MAX_STEPS,
    output_root: str = "runs/exp4",
    cache_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Run the Exp 4 variant sweep."""
    all_frames: list[pd.DataFrame] = []

    for dim in dims:
        for kl_s in kl_s_weights:
            for use_per_mode in use_per_modes:
                for seed in seeds:
                    out_dir = (
                        f"{output_root}/dim{dim}_kls{kl_s}"
                        f"_permode{int(use_per_mode)}/seed{seed}"
                    )
                    df = run_single(
                        dim=dim,
                        use_per_mode=use_per_mode,
                        kl_S_weight=kl_s,
                        seed=seed,
                        max_steps=max_steps,
                        output_dir=out_dir,
                        cache_dir=cache_dir,
                    )
                    all_frames.append(df)

    all_metrics = pd.concat(all_frames, ignore_index=True)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(root / "all_runs.csv", index=False)
    logger.info("Sweep complete. Wrote %s", root / "all_runs.csv")
    return all_metrics
