"""
Phase 3 — ESDM-modified VibrationalStateBlock.

Wraps the VDT ``VibrationalStateBlock`` from ``esdm._vendor.vdt`` and
replaces the fixed VDT time step with the entropic quantities supplied by
``EntropicClock``:

    Q_next = 2Q - Q_prev
           - delta_tau² · H(Q)          [hybridisation operator]
           - γ · ΔQ
           + Lambda_tau · delta_tau² · B_t

The hybridisation operator H = Π ∘ Diffuse_LF ∘ Φ (issue #20) replaces the
bare ``L_f @ Q_t`` product with a three-step spectral procedure:
  1. Lifting Φ: project Q onto the U_q eigenvector subspace
  2. Laplacian diffusion: parabolic heat step along the feature-space graph
  3. Rayleigh gradient projection Π: convert diffused signal to a restoring
     force via the Rayleigh quotient gradient

The wrapper reshapes the ESDM input ``x: (B, F)`` to a single-channel
node-feature state ``Q: (B, F, 1)`` so that the feature-space Laplacian
``L_f: (F, F)`` acts on the node dimension, exactly as the design formula
requires.  The previous state ``Q_tm1`` is kept as a buffer so the wave
recurrence can persist across forward calls.

When ``U_q`` is not supplied to ``forward()``, the block falls back to the
bare ``L_f @ Q_t`` product (backward-compatible with the original VDT).

See docs/00-DESIGN.md §4 for the full hybridisation operator derivation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .._vendor.laplacian import DifferentiableLaplacian
from .._vendor.vdt import VibrationalStateBlock as _VDTVibrationalBlock


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
    VDT CFL helper has a base Laplacian to fall back to.

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
    sigma_sq : float
        Initial value of the diffusion coefficient sigma^2 for the Laplacian
        heat step in the hybridisation operator.
    eta : float
        Initial value of the Rayleigh gradient step size eta for the
        projection step Pi in the hybridisation operator.
    freeze_h : bool
        If True, sigma^2 and eta are frozen (not trainable) -- used for the
        frozen-vs-learned ablation (issue #20, scope item 4).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_steps: int = 4,
        gamma: float = 0.1,
        dropout: float = 0.1,
        density_momentum: float = 0.9,
        sigma_sq: float = 1.0,
        eta: float = 0.1,
        freeze_h: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.freeze_h = freeze_h

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

        # Hybridisation operator parameters (issue #20).
        self.sigma_sq = nn.Parameter(torch.tensor(float(sigma_sq)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        if freeze_h:
            self.sigma_sq.requires_grad_(False)
            self.eta.requires_grad_(False)

        # Previous state buffer, shape (B, F, 1).  Resized lazily per batch.
        self.register_buffer("Q_tm1", torch.zeros(0))

        # Diagnostics buffers for net_energy_flow and pi_contraction_ratio.
        self.register_buffer("_last_angular_displacement", torch.tensor(0.0))
        self.register_buffer("_last_pi_contraction_ratio", torch.tensor(0.0))

    # ------------------------------------------------------------------ #

    def _ensure_Q_tm1(self, Q_t: Tensor) -> Tensor:
        """Return a Q_tm1 tensor matching ``Q_t`` shape, resizing the buffer."""
        if self.Q_tm1.shape != Q_t.shape:
            self.Q_tm1 = torch.zeros_like(Q_t)
        return self.Q_tm1

    # ------------------------------------------------------------------ #

    def _hybridisation(
        self,
        Q_t: Tensor,
        L_f: Tensor,
        U_q: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """
        Full hybridisation operator H = Pi . Diffuse_LF . Phi.

        Returns the generalised Laplacian action (analogous to L*Q) as
        (B, F, 1).  The wave recurrence subtracts dt^2 * H_action to
        produce the restoring force, exactly as the bare L_f @ Q_t term
        does in the non-hybridised case.

        The design doc (00-DESIGN.md §4.4) defines Pi = -eta * grad_R,
        but the wave recurrence has -dt^2 * Pi which would give +eta*dt^2*grad_R
        (destabilising).  The physically correct sign keeps H_action =
        eta * grad_R (analogous to L*Q), so -dt^2 * H_action is the
        restoring force toward low-energy modes.

        Parameters
        ----------
        Q_t : Tensor (B, F, 1)
            Current vibrational state.
        L_f : Tensor (F, F) or (B, F, F)
            Feature-space Laplacian.
        U_q : Tensor (F, q)
            Leading eigenvectors of L_F from SpectralLoadingDecoder.
        dt : Tensor, scalar
            CFL-safe time step (first power, not squared).
        """
        Q_2d = Q_t.squeeze(-1)  # (B, F)
        L = L_f[0] if L_f.dim() == 3 else L_f  # (F, F)

        # Step 1 — Lifting Phi: F_t = Q_t @ U_q @ U_q^T
        P = U_q @ U_q.t()  # (F, F) projection matrix
        F_t = Q_2d @ P  # (B, F)

        # Step 2 — Laplacian diffusion: F_tilde = F_t - (sigma^2 * dt / 2) * (F_t @ L_F)
        L_F_F_t = F_t @ L  # (B, F)
        F_tilde = F_t - (self.sigma_sq * dt / 2.0) * L_F_F_t  # (B, F)

        # Step 3 — Rayleigh gradient projection Pi
        L_F_F_tilde = F_tilde @ L  # (B, F)

        # Row-wise Rayleigh quotient: R_i = f_i^T L f_i / f_i^T f_i
        norms_sq = (F_tilde * F_tilde).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        R = (F_tilde * L_F_F_tilde).sum(dim=-1, keepdim=True) / norms_sq  # (B, 1)

        # Rayleigh gradient: (2 / ||F_tilde||^2) * (L_F F_tilde - R * F_tilde)
        grad_R = (2.0 / norms_sq) * (L_F_F_tilde - R * F_tilde)  # (B, F)

        # Project through lifting subspace
        grad_R_projected = grad_R @ P  # (B, F)

        # H action = eta * grad_R (analogous to L*Q)
        H_action = self.eta * grad_R_projected  # (B, F)

        # Track pi_contraction_ratio = ||Delta Q|| / ||Q||
        with torch.no_grad():
            pi_norm = H_action.norm(dim=-1, keepdim=True)
            q_norm = Q_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self._last_pi_contraction_ratio.fill_(float((pi_norm / q_norm).mean().item()))

        return H_action.unsqueeze(-1)  # (B, F, 1)

    # ------------------------------------------------------------------ #

    def _step(
        self,
        Q_t: Tensor,
        Q_tm1: Tensor,
        L_f: Tensor,
        lap: DifferentiableLaplacian,
        Lambda_tau: float | Tensor,
        delta_tau: float | Tensor,
        U_q: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Execute one ESDM wave recurrence step using the VDT mixing block.

        All tensors are 3-D: ``(B, F, 1)``.

        When ``U_q`` is provided, the hybridisation operator H replaces the
        bare ``L_f @ Q_t`` term.  When ``U_q`` is None, falls back to the
        bare Laplacian product (backward-compatible with original VDT).
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

        # Laplacian / hybridisation term
        if U_q is not None:
            L_term = self._hybridisation(Q_t, L_f, U_q, dt)
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
        U_q: Optional[Tensor] = None,       # (F, q) leading eigenvectors of L_F
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns (Q_t, rho_plus, rho_minus).

        ``Q_t`` has the same shape as the input ``x`` (i.e. ``(B, F)``).
        ``rho_plus`` and ``rho_minus`` are ``(F, F)`` PSD density matrices.

        When ``U_q`` is provided, the full hybridisation operator H is
        applied in each wave recurrence step.  When ``U_q`` is None, the
        bare ``L_f @ Q_t`` product is used (backward-compatible).
        """
        if x.dim() != 2:
            raise ValueError(f"VibrationalStateBlock expects x of shape (B, F). Got {tuple(x.shape)}.")
        if L_f.shape[-2:] != (x.shape[1], x.shape[1]):
            raise ValueError(
                f"L_f shape {tuple(L_f.shape)} incompatible with x feature dim {x.shape[1]}."
            )
        if U_q is not None and U_q.shape[0] != x.shape[1]:
            raise ValueError(
                f"U_q shape {tuple(U_q.shape)} incompatible with x feature dim {x.shape[1]}."
            )

        # ESDM (B, F) -> VDT (B, F, 1)
        Q_t = x.unsqueeze(-1)
        # Reset Q_tm1 to Q_t at the start of each forward call so the wave
        # recurrence starts from rest (zero initial velocity). Carrying Q_tm1
        # from the previous batch injects energy through the 2*Q_t - Q_tm1
        # inertial term because Q_tm1 is from a different input batch.
        Q_tm1 = Q_t.clone()
        lap = _laplacian_from_dense(L_f)

        # Apply n_steps recurrence steps, carrying Q_tm1 through.
        for _ in range(self.n_steps):
            Q_tp1 = self._step(Q_t, Q_tm1, L_f, lap, Lambda_tau, delta_tau, U_q=U_q)
            Q_tm1 = Q_t
            Q_t = Q_tp1

        # Measure direction only. The half-angle formula resolves small rotations
        # without the loss of precision of acos near one.
        with torch.no_grad():
            start = F.normalize(x, dim=-1)
            finish = F.normalize(Q_t.squeeze(-1), dim=-1)
            angles = 2 * torch.atan2((start - finish).norm(dim=-1), (start + finish).norm(dim=-1))
            self._last_angular_displacement.copy_(angles.mean())

        self.Q_tm1 = Q_tm1.detach().clone()

        rho_plus = self._vdt_block.density.rho_plus
        rho_minus = self._vdt_block.density.rho_minus
        return Q_t.squeeze(-1), rho_plus, rho_minus

    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict[str, float]:
        """Return the last recorded H-operator diagnostics."""
        return {
            "angular_displacement": float(self._last_angular_displacement),
            "pi_contraction_ratio": float(self._last_pi_contraction_ratio),
            "gamma_mean": float(self._vdt_block.gamma.detach().mean()),
        }
