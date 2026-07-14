r"""
Latent-space evaluation for ESDM real-data experiments (issue #22).

Metrics:
- Reconstruction MSE on a held-out test batch.
- Conservation drift: mean absolute deviation of total entropy from S_0.
- Downstream k-NN accuracy in the latent space (uses encoder mu as embedding).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor


def reconstruction_mse(x: Tensor, x_hat: Tensor) -> float:
    """Mean squared error between input and reconstruction."""
    return float(torch.nn.functional.mse_loss(x_hat, x).detach())


def conservation_drift(model, x: Tensor) -> float:
    """
    Compute conservation drift for a batch.

    Returns mean |S_total - S_0| / (|S_0| + eps).
    """
    with torch.no_grad():
        _ = model(x)
        # S_total is logged in the model's health_check as 'conserved'.
        # We approximate it by the clock's conserved value.
        total = torch.tensor(model.health_check()["conserved"], dtype=x.dtype, device=x.device)
        s0 = model.S_0 if model.S_0 is not None else total
        drift = (total - s0).abs() / (s0.abs() + 1e-8)
        return float(drift.detach())


def knn_latent_accuracy(
    model,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    k: int = 5,
) -> float:
    """
    Train a k-NN classifier on encoder latent means and report accuracy.

    Parameters
    ----------
    model : ESDM
        Trained model with ``enc_mu`` encoder.
    x_train, x_test : Tensor
        Input embeddings.
    y_train, y_test : Tensor or ndarray
        Labels (any integer type).
    """
    model.eval()
    with torch.no_grad():
        z_train = model.enc_mu(x_train).cpu().numpy()
        z_test = model.enc_mu(x_test).cpu().numpy()
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    clf = KNeighborsClassifier(n_neighbors=min(k, len(y_train)))
    clf.fit(z_train, y_train)
    return float(clf.score(z_test, y_test))


def evaluate_latent(
    model,
    loader,
    n_test: int = 256,
    seed: int = 3407,
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate a trained ESDM model on a real-data loader.

    Parameters
    ----------
    model : ESDM
    loader :
        Data loader with ``test_batch`` and test labels (``_test_targets`` or
        ``_test_labels``).
    n_test : int
        Number of test samples to use.
    """
    model.eval()
    x_test = loader.test_batch(n_test, seed=seed)
    with torch.no_grad():
        out = model(x_test)

    results = {
        "reconstruction_mse": reconstruction_mse(x_test, out.x_hat),
        "conservation_drift": conservation_drift(model, x_test),
    }

    # k-NN latent accuracy if labels are available.
    labels = getattr(loader, "_test_targets", None)
    if labels is None:
        labels = getattr(loader, "_test_labels", None)
    if labels is not None and len(labels) > 0:
        x_train = loader.train_batch(min(n_test * 2, len(labels)), seed=seed + 1)
        train_idx = torch.randperm(len(labels))[:n_test * 2]
        y_train = labels[train_idx]
        test_idx = torch.randperm(len(labels))[:n_test]
        y_test = labels[test_idx]
        x_test_labeled = loader.test_data[test_idx]
        results["knn_accuracy"] = knn_latent_accuracy(
            model, x_train, y_train, x_test_labeled, y_test, k=k
        )

    model.train()
    return results
