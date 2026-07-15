"""
Dataset adapter protocol for ESDM.

Defines the interface that all dataset loaders must implement to provide
dataset-specific training constants. The model code (esdm/models/) is
completely dataset-agnostic; only the adapter and experiment code know
which dataset is being used.

The adapter extends the existing loader interface (dim_input, train_data,
train_batch, etc.) with training constants that depend on the data
distribution:

- eps_rel: relative heat-death threshold. Derived from the irreducible
  batch-level KL variance, which is dataset-specific. On synthetic data the
  KL is deterministic (eps_rel=0); on real stochastic batches the KL has
  irreducible variance ~O(0.01) that must not trigger false heat death.
- default_lr: the learning rate that produces stable training. Depends on
  the embedding scale and dimensionality.
- default_n_steps: the step count at which heat death typically fires.
  Depends on the convergence rate of the encoder on this data.

These are NOT model hyperparameters -- they are data characteristics. The
model's architecture constants (gamma, sigma_sq, eta, q, etc.) are defined
in 001-IMPLEMENTATION-PLAN.md §2 and are dataset-agnostic.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol that all ESDM dataset loaders must satisfy."""

    @property
    def dim_input(self) -> int:
        """Embedding dimension F."""
        ...

    @property
    def train_data(self) -> Tensor:
        """Full train split, shape (N, F)."""
        ...

    @property
    def test_data(self) -> Tensor:
        """Full test split, shape (N_test, F)."""
        ...

    def train_batch(self, batch_size: int, seed: Optional[int] = None) -> Tensor:
        """Random batch from the train split, shape (batch_size, F)."""
        ...

    def test_batch(self, batch_size: int, seed: Optional[int] = None) -> Tensor:
        """Random batch from the test split, shape (batch_size, F)."""
        ...

    @property
    def eps_rel(self) -> float:
        """
        Relative heat-death threshold for the entropic clock.

        0.0 for synthetic data (KL is deterministic).
        ~1e-3 for real stochastic data (irreducible batch KL variance).
        """
        ...

    @property
    def default_lr(self) -> float:
        """Learning rate that produces stable training on this data."""
        ...

    @property
    def default_n_steps(self) -> int:
        """Step count at which heat death typically fires on this data."""
        ...

    @property
    def graph_params(self) -> dict:
        """
        ArrowSpace tuner-discovered graph parameters for this dataset.

        Cached after a single tuner run. Contains:
        - eps: epsilon for ArrowSpace lambda graph
        - k: k-NN parameter
        - tau: tau mode parameter

        If empty dict, the tuner runs on first use and caches here.
        """
        ...

    @property
    def heat_death_patience(self) -> int:
        """
        Number of consecutive flat-clock steps before declaring heat death.

        Dataset-specific: depends on batch KL variance and convergence rate.
        """
        ...
