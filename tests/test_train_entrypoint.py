"""Exercise the training entry point with the dtype emitted by the data loaders."""

import torch

from esdm.cli import train


def test_train_uses_model_dtype_for_dataset_batches(monkeypatch):
    class Loader:
        def train_batch(self, batch_size, seed):
            rng = torch.Generator().manual_seed(seed)
            return torch.randn(batch_size, 8, generator=rng, dtype=torch.float64)

    monkeypatch.setattr(train, "_build_loader", lambda dataset, seed: (Loader(), 8))
    train.main(["--dataset", "mnist", "--steps", "2", "--seed", "3407"])
