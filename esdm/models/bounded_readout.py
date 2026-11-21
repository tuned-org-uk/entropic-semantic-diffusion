"""Bounded modal gains on the frozen ESDM reconstruction chart."""
from __future__ import annotations

from math import isqrt

import numpy as np
from scipy.special import expi
import torch
from torch import Tensor, nn


class BoundedSpectralReadout(nn.Module):
    """Scale the retained modes and transmit the orthogonal complement unchanged."""

    def __init__(self, q: int) -> None:
        super().__init__()
        if q < 1:
            raise ValueError("q must be positive")

        # The v2 benchmark spans four doublings above Schoenfeld's threshold.
        x = np.geomspace(4096, 65536, q).round().astype(np.int64)
        end = int(x[-1])
        prime = np.ones(end + 1, dtype=bool)
        prime[:2] = False
        for p in range(2, isqrt(end) + 1):
            if prime[p]:
                prime[p * p :: p] = False
        count = prime.cumsum()[x]
        log_x = np.log(x)
        li = expi(log_x)
        bound = np.sqrt(x) * log_x / (8 * np.pi)
        residual = (count - li) / bound
        if not np.all(np.abs(residual) < 1):
            raise ValueError("prime-count residual outside the open Schoenfeld interval")

        dtype = torch.get_default_dtype()
        # Deterministic constants move with the module; a checkpoint cannot replace them.
        self.register_buffer("radius", torch.tensor(bound / li, dtype=dtype), persistent=False)
        self.register_buffer("initial_weights", torch.tensor(count / li, dtype=dtype),
                             persistent=False)
        self.logits = nn.Parameter(torch.tensor(np.arctanh(residual), dtype=dtype))

    def weights(self) -> Tensor:
        """Return the constrained gain for each chart column in ascending order."""
        return 1 + self.radius * self.logits.tanh()

    def forward(self, Q: Tensor, U_q: Tensor) -> Tensor:
        """Read row-vector signals Q in a frozen orthonormal chart U_q with shape (F, q)."""
        q = self.logits.numel()
        if Q.ndim < 1 or U_q.shape != (Q.shape[-1], q) or q > Q.shape[-1]:
            raise ValueError("U_q must have shape (Q.shape[-1], q) with q <= feature dimension")
        return Q + ((Q @ U_q) * (self.weights() - 1)) @ U_q.T
