"""
Phase 3 — ESDM-modified VibrationalStateBlock.

Wraps the VDT ``VibrationalStateBlock`` from ``esdm._vendor.vdt`` and
replaces the fixed VDT time step with the entropic quantities supplied by
``EntropicClock``:

    Q_next = 2Q - Q_prev
           - delta_tau² · L_f @ Q
           - γ · ΔQ
           + Lambda_tau · delta_tau² · B_t

The wrapper reshapes the ESDM input ``x: (B, F)`` to a single-channel
node-feature state ``Q: (B, F, 1)`` so that the feature-space Laplacian
``L_f: (F, F)`` acts on the node dimension, exactly as the design formula
requires.  The previous state ``Q_tm1`` is kept as a buffer so the wave
recurrence can persist across forward calls.

See docs/01-IMPLEMENTATION-PLAN.md §4 for the full specification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .._vendor.laplacian import DifferentiableLaplacian
from .._vendor.vdt import VibrationalStateBlock as _VDTVibrationalBlock
from ..graphs.wht_filter import GraphWHTFilter


def compute_mode_residuals(
    Q_t: Tensor,
    Q_prev: Tensor,
    bright_modes: Optional[int] = None,
) -> Tensor:
    """
    Per-mode residual between two vibrational states.

    Returns ``r_k = ||Q_{t,k} - Q_{t-1,k}||_2`` for each feature mode ``k``
    (L2 norm taken over the batch dimension).  The result is a 1-D tensor of
    length ``F``; non-negative by construction.

    Parameters
    ----------
    Q_t : Tensor, shape (B, F) or (B, F, 1)
        Current vibrational state.
    Q_prev : Tensor, shape matching ``Q_t``
        Previous vibrational state.
    bright_modes : int, optional
        If provided, only the top-``bright_modes`` residuals are retained;
        inactive modes are forced to zero.  This lets the residual signal match
        the bright/dark sector split used by the entropic clock.
    """
    if Q_t.dim() == 3:
        Q_t = Q_t.squeeze(-1)
    if Q_prev.dim() == 3:
        Q_prev = Q_prev.squeeze(-1)

    if Q_t.shape != Q_prev.shape:
        raise ValueError(
            f"Q_t shape {tuple(Q_t.shape)} does not match Q_prev shape "
            f"{tuple(Q_prev.shape)}"
        )

    diff = Q_t - Q_prev
    r = diff.norm(dim=0)

    if bright_modes is not None and 0 < bright_modes < r.shape[0]:
        order = torch.argsort(r, descending=True)
        mask = torch.zeros_like(r, dtype=torch.bool)
        mask[order[:bright_modes]] = True
        r = r * mask

    return r


def _laplacian_from_dense(L: Tensor, eps: float = 1e-8) -> DifferentiableLaplacian:
    """
    Build a ``DifferentiableLaplacian`` from a dense Laplacian matrix so the
    VDT CFL helper has a base Laplacian to fall back on.

    Parameters
    ----------
    L : Tensor  (F, F) or (B, F, F)
        Dense feature-space Laplacian.  The first batch element is used if
        a batch is supplied.
    eps : float
        Threshold below which an off-diagonal entry is treated as zero.

    Returns
    -------
    DifferentiableLaplacian with ``n_nodes = F``.
    """
    if L.dim() == 3:
        L = L[0]
    F = L.shape[0]
    # Recover an adjacency from the Laplacian: A = -L off-diagonal.
    A = -L
    A = A - torch.diag(torch.diag(A))  # zero diagonal
    mask = A > eps
    rows, cols = torch.where(mask)
    edge_index = torch.stack([rows, cols], dim=0)
    base_weights = A[rows, cols]
    if edge_index.numel() == 0:
        # Degenerate case: no edges. Create a self-loop so the object is valid.
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=L.device)
        base_weights = torch.tensor([0.0], dtype=L.dtype, device=L.device)
    return DifferentiableLaplacian(
        n_nodes=F,
        edge_index=edge_index,
        base_weights=base_weights,
        normalised=False,
        sparse=False,
    )


def _cfl_dt(
    log_dt_free: Tensor,
    L_f: Tensor,
    lap: DifferentiableLaplacian,
) -> Tensor:
    """
    Return a CFL-safe time step matching VDT's ``_cfl_dt_Lf`` logic.

    Combines the learnable free step, the Gershgorin bound computed from
    the dynamic ``L_f``, and the frozen base-Laplacian bound from ``lap``,
    and returns the strictest (smallest) of the three.
    """
    dt_free = log_dt_free.exp()

    # Gershgorin bound on lambda_max(L_f) for the first batch element.
    L_single = L_f[0] if L_f.dim() == 3 else L_f
    lam_gersh = L_single.abs().sum(dim=-1).max().clamp(min=1e-8)
    dt_lf = (2.0 / lam_gersh).sqrt().to(dtype=dt_free.dtype, device=dt_free.device)

    # Frozen base-Laplacian bound.
    dt_base = torch.tensor(
        lap.dt_max_cfl(), dtype=dt_free.dtype, device=dt_free.device
    )

    return torch.minimum(torch.minimum(dt_free, dt_lf), dt_base)


class VibrationalStateBlock(nn.Module):
    """
    ESDM wrapper around the VDT ``VibrationalStateBlock``.

    The VDT block operates on ``(B, N, d)`` node-feature states.  This
    wrapper maps the ESDM input ``x: (B, F)`` to ``Q_t: (B, F, 1)`` (F nodes,
    1 feature channel) so that the feature-space Laplacian ``L_f: (F, F)``
    acts directly on the node dimension.  The previous state ``Q_tm1`` is
    stored as a buffer so the recurrence can continue across forward calls.

    Parameters
    ----------
    dim : int
        Feature dimension ``F`` of the ESDM input.
    n_heads : int
        Attention heads in the transformer mixing block.  When ``dim == 1``
        this is forced to ``1`` because the feature channel is a single
        scalar.
    n_steps : int
        Number of internal VDT recurrence steps to apply per forward call.
    gamma : float
        Initial value of the per-feature Rayleigh damping vector.
    dropout : float
        Dropout probability inside the mixing block.
    density_momentum : float
        EMA momentum forwarded to the VDT ``SignedDensityMatrix`` update.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_steps: int = 4,
        gamma: float = 0.1,
        dropout: float = 0.1,
        density_momentum: float = 0.9,
        wht_filter: Optional["GraphWHTFilter"] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.wht_filter = wht_filter

        # The feature channel is 1: x (B, F) -> Q (B, F, 1).
        # MultiheadAttention requires embed_dim divisible by num_heads.
        # feat_dim is always 1 in this wrapper, so num_heads must be 1.
        effective_heads = 1

        self._vdt_block = _VDTVibrationalBlock(
            n_nodes=dim,
            feat_dim=1,
            n_heads=effective_heads,
            dropout=dropout,
            density_momentum=density_momentum,
        )

        # Honour the wrapper gamma initial value.  VDT stores a raw parameter
        # such that gamma = softplus(raw).  softplus^{-1}(g) = log(exp(g) - 1).
        if gamma > 0.0:
            raw = torch.log(torch.exp(torch.tensor(gamma)) - 1.0)
            with torch.no_grad():
                self._vdt_block._gamma_raw.fill_(raw)

        # Previous state buffer, shape (B, F, 1).  Resized lazily per batch.
        self.register_buffer("Q_tm1", torch.zeros(0))

    # ------------------------------------------------------------------ #

    def _ensure_Q_tm1(self, Q_t: Tensor) -> Tensor:
        """Return a Q_tm1 tensor matching ``Q_t`` shape, resizing the buffer."""
        if self.Q_tm1.shape != Q_t.shape:
            self.Q_tm1 = torch.zeros_like(Q_t)
        return self.Q_tm1

    # ------------------------------------------------------------------ #

    def _wht_filtered_laplacian_action(
        self,
        L_f: Tensor,
        Q_t: Tensor,
    ) -> Tensor:
        """
        Compute the WHT-filtered Laplacian action inside the wave recurrence.

        Implements:  L̃_f Q_t = H_F^T (g ⊙ (L̂_f · Q̂_t^T)^T)

        where:
          H_F   = normalised Hadamard matrix (F×F, precomputed)
          L̂_f   = H_F L_f H_F^T  (WHT-domain Laplacian)
          Q̂_t   = H_F Q_t         (WHT of the state)
          g     = spectral gate vector (heat-kernel or bivariate)

        This is the Phase 2 integration from docs/00-DESIGN.md §4.3.
        """
        F = L_f.shape[-1]
        wht = self.wht_filter

        # Build normalised Hadamard matrix H_F in torch.
        # Uses the Fiedler ordering from GraphWHTFilter for sequency alignment.
        if not hasattr(self, '_H_F') or self._H_F.shape[0] != F:
            import numpy as np
            P = wht.P if wht.P is not None else F
            # Build Hadamard matrix of size P×P, then truncate to F×F.
            H = np.eye(P)
            h = 1
            while h < P:
                for i in range(0, P, h * 2):
                    for j in range(i, i + h):
                        a, b = H[j].copy(), H[j + h].copy()
                        H[j] = a + b
                        H[j + h] = a - b
                h *= 2
            H = H / np.sqrt(P)
            # Apply Fiedler ordering to rows/cols
            if wht.fiedler_order is not None:
                order = wht.fiedler_order[:F]
                H_ordered = np.eye(P)
                H_ordered[:F, :F] = H[np.ix_(order, order)]
                H = H_ordered
            self._H_F = torch.from_numpy(H[:F, :F].astype(np.float32)).to(device=L_f.device, dtype=L_f.dtype).detach()
            self._H_F_T = self._H_F.t().detach()

        H_F = self._H_F
        H_F_T = self._H_F_T

        # L̂_f = H_F L_f H_F^T (precomputed lazily)
        if not hasattr(self, '_L_hat') or self._L_hat.shape != L_f.shape:
            with torch.no_grad():
                if L_f.dim() == 3:
                    self._L_hat = torch.bmm(torch.bmm(H_F.unsqueeze(0).expand_as(L_f), L_f), H_F_T.unsqueeze(0).expand_as(L_f))
                else:
                    self._L_hat = H_F @ L_f @ H_F_T
        L_hat = self._L_hat

        # Q̂_t = H_F Q_t (apply Hadamard to the feature dimension)
        Q_hat = torch.bmm(H_F.unsqueeze(0).expand(Q_t.shape[0], -1, -1), Q_t) if Q_t.dim() == 3 else H_F @ Q_t

        # Gate g: heat-kernel for basic mode, bivariate for bivariate mode.
        # wht_filter.a_tau_scale and wht_filter.tau_decay_power control the
        # bivariate gate. The gate attenuates high-sequency modes.
        import numpy as np
        tau_t = float(wht.tau_max_est) if hasattr(wht, 'tau_max_est') else 1.0
        if hasattr(wht, 'wht_lambdas') and wht.wht_lambdas is not None:
            wht_lam = wht.wht_lambdas[:F]
            # Heat-kernel gate: g_k = exp(-t * λ_k)
            g_np = np.exp(-tau_t * wht_lam)
            g = torch.from_numpy(g_np.astype(np.float32)).to(device=L_f.device, dtype=L_f.dtype).detach()
        else:
            g = torch.ones(F, dtype=L_f.dtype, device=L_f.device).detach()

        # Filtered action: H_F^T (g ⊙ (L̂_f · Q̂_t))
        LQ_hat = torch.bmm(L_hat, Q_hat) if L_hat.dim() == 3 else torch.matmul(L_hat, Q_hat)
        gated = g.unsqueeze(0).unsqueeze(-1) * LQ_hat if LQ_hat.dim() == 3 else g * LQ_hat
        L_tilde = torch.bmm(H_F_T.unsqueeze(0).expand(Q_t.shape[0], -1, -1), gated) if gated.dim() == 3 else H_F_T @ gated

        return L_tilde

    def _step(
        self,
        Q_t: Tensor,
        Q_tm1: Tensor,
        L_f: Tensor,
        lap: DifferentiableLaplacian,
        Lambda_tau: float | Tensor,
        delta_tau: float | Tensor,
    ) -> Tensor:
        """
        Execute one ESDM wave recurrence step using the VDT mixing block.

        All tensors are 3-D: ``(B, F, 1)``.
        """
        block = self._vdt_block

        # Transformer mixing B_t = FFN(MHA(Q_t))
        h, _ = block.attn(block.norm1(Q_t), block.norm1(Q_t), block.norm1(Q_t))
        Q_mix = Q_t + h
        B_t = Q_mix + block.ffn(block.norm2(Q_mix))

        # CFL-safe entropic time step
        delta_tau_t = (
            torch.tensor(delta_tau, dtype=Q_t.dtype, device=Q_t.device)
            if isinstance(delta_tau, float) else delta_tau.to(dtype=Q_t.dtype, device=Q_t.device)
        )
        dt = _cfl_dt(block.log_dt, L_f, lap)
        # If a smaller delta_tau is requested, use it; otherwise clamp to CFL.
        dt = torch.minimum(delta_tau_t, dt)
        dt2 = dt * dt

        # Wave update: Δτ² · L_f @ Q_t (WHT-filtered when a filter is injected)
        if self.wht_filter is not None:
            L_term = self._wht_filtered_laplacian_action(L_f, Q_t)
        else:
            L_term = torch.bmm(L_f, Q_t) if L_f.dim() == 3 else torch.matmul(L_f, Q_t)
        damp = block.gamma.view(1, 1, -1) * (Q_t - Q_tm1)

        Lambda_tau_t = (
            torch.tensor(Lambda_tau, dtype=Q_t.dtype, device=Q_t.device)
            if isinstance(Lambda_tau, float) else Lambda_tau.to(dtype=Q_t.dtype, device=Q_t.device)
        )

        Q_tp1 = (
            2.0 * Q_t - Q_tm1
            - dt2 * L_term
            - damp
            + Lambda_tau_t * dt2 * B_t
        )

        # Wire the signed density to the consecutive wave states.
        block.density.update(Q_t, Q_tp1, alpha=block.density_momentum)
        return Q_tp1

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,                          # (B, F)
        L_f: Tensor,                        # (F, F) feature-space Laplacian
        Lambda_tau: float | Tensor,         # entropic pump from EntropicClock
        delta_tau: float | Tensor,          # entropic time step from EntropicClock
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns (Q_t, rho_plus, rho_minus).

        ``Q_t`` has the same shape as the input ``x`` (i.e. ``(B, F)``).
        ``rho_plus`` and ``rho_minus`` are ``(F, F)`` PSD density matrices.
        """
        if x.dim() != 2:
            raise ValueError(f"VibrationalStateBlock expects x of shape (B, F). Got {tuple(x.shape)}.")
        if L_f.shape[-2:] != (x.shape[1], x.shape[1]):
            raise ValueError(
                f"L_f shape {tuple(L_f.shape)} incompatible with x feature dim {x.shape[1]}."
            )

        # ESDM (B, F) -> VDT (B, F, 1)
        Q_t = x.unsqueeze(-1)
        Q_tm1 = self._ensure_Q_tm1(Q_t)
        lap = _laplacian_from_dense(L_f)

        # Apply n_steps recurrence steps, carrying Q_tm1 through.
        for _ in range(self.n_steps):
            Q_tp1 = self._step(Q_t, Q_tm1, L_f, lap, Lambda_tau, delta_tau)
            Q_tm1 = Q_t
            Q_t = Q_tp1

        self.Q_tm1 = Q_tm1.detach().clone()

        rho_plus = self._vdt_block.density.rho_plus
        rho_minus = self._vdt_block.density.rho_minus
        return Q_t.squeeze(-1), rho_plus, rho_minus
