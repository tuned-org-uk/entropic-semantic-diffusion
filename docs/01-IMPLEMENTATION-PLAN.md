# 01 — ESDM: PyTorch Implementation Plan

> **Status:** Implementation-ready design  
> **Depends on:** `docs/00-DESIGN.md`, `000-DIFFUSION-ON-PHYSICAL-SURFACES.md`  
> **Integrates:** ArrowSpace dual-space fusion analysis, Barontini 2026, SPIN/GW spectral indexing, 2.5-D hypothesis  
> **Target:** PyTorch ≥ 2.3, Python ≥ 3.11

---

## 0. Scope and Constraints

This plan translates the conceptual ESDM design into an ordered sequence of self-contained,
testable PyTorch modules. Each phase ends with a runnable unit test so the build never
breaks without an explicit checkpoint.

**Constraints fixed by design:**
- No external time variable anywhere in the forward pass after Phase 2.
- The feature-space Laplacian `L_F ∈ R^{F×F}` is built from the ArrowSpace GW algorithm
  (eigenmaps mode only; energymaps is out of scope).
- The fusion space is **item-space** because both training and inference operate there.
- The fused matrix is therefore fixed to **Mode B**:
  `M_N = α·norm(VV^T) − β·norm(VL_F V^T)`.
- The dual-space fused matrix is computed once per training run and cached; it is not
  re-computed per batch.
- `S_0` is fixed from the **first batch** as `kl_tau(0) + dark_entropy(0)` and then stored
  as a non-gradient buffer.

---

## 1. Repository Layout

```
enthropy-semantic-diffusion/
├── docs/
│   ├── 00-DESIGN.md
│   ├── 00-DESIGN_PREAMBLE.md
│   └── 01-IMPLEMENTATION-PLAN.md          ← this file
├── esdm/
│   ├── __init__.py
│   ├── clock.py                            # EntropicClock
│   ├── laplacian.py                        # Item-space fused matrix M_N
│   ├── vibrational.py                      # VibrationalStateBlock (modified)
│   ├── spectral_decoder.py                 # SpectralLoadingDecoder (unchanged interface)
│   ├── diffusion_decoder.py                # EntropicDiffusionDecoder
│   ├── elbo.py                             # ESDM ELBO with conservation term
│   └── model.py                            # Top-level ESDM nn.Module
├── tests/
│   ├── test_clock.py
│   ├── test_laplacian.py
│   ├── test_vibrational.py
│   ├── test_elbo.py
│   └── test_model.py
├── scripts/
│   ├── build_laplacian.py                  # one-shot: compute and save L_F and M_N
│   ├── train.py
│   └── sweep_kl_s.py                       # barrier-height sweep (Section 9 of design)
└── requirements.txt
```

---

## 2. Phase 1 — Item-Space Fused Matrix `M_N` (`esdm/laplacian.py`)

This is the foundational data structure identified in the dual-space analysis. It must be
built before any forward pass and is the only fusion matrix used by training and inference.

### 2.1 Mathematical object

Let `V ∈ R^{N×F}` be the corpus embedding matrix and `L_F ∈ R^{F×F}` the feature-space
graph Laplacian built by the ArrowSpace GW algorithm on the *transposed* feature matrix.

The fused item-space kernel is:

```text
M_N = α·norm(VV^T) − β·norm(VL_F V^T)
```

Interpretation:
- `VV^T` is geometric item-item affinity.
- `VL_FV^T` is semantic roughness/interaction induced by the feature graph.
- The subtraction makes the semantic term act as a structural penalty on purely geometric
  neighborhoods, consistent with the SPIN/ArrowSpace reading of Laplacian energy.

### 2.2 `laplacian.py` API

```python
class DualSpaceMatrix:
    """
    Builds and caches the fused item-space geometric-semantic matrix.

    Parameters
    ----------
    V : Tensor  (N, F)   — corpus embeddings
    L_F : Tensor  (F, F) — feature-space Laplacian from ArrowSpace GW
    alpha : float        — weight on geometric term (default 0.5)
    beta : float         — weight on semantic term  (default 0.5)

    Attributes
    ----------
    M : Tensor            — fused item-space matrix M_N (cached)
    eigvals : Tensor      — eigenvalues of M_N (for health monitoring)
    eigvecs : Tensor      — eigenvectors of M_N (for item-space projection)
    fiedler_vec : Tensor  — second eigenvector of M_N (semantic universe size proxy)
    """

    def build(self, V: Tensor, L_F: Tensor) -> None: ...

    def project(self, x: Tensor, k: int) -> Tensor:
        """Project item vectors onto top-k eigenvectors of M_N."""
        ...

    def rayleigh(self, x: Tensor) -> Tensor:
        """R(x) = x^T M_N x / x^T x  — fused Dirichlet energy in item-space."""
        ...
```

### 2.3 Normalization convention

Both `VV^T` and `VL_F V^T` are normalized by their Frobenius norm before blending.
This ensures `alpha` and `beta` are dimensionless mixing coefficients analogous to the
`tau` parameter in ArrowSpace SPIN.

### 2.4 Test checkpoint

`tests/test_laplacian.py`: for random `V (50, 32)` and `L_F (32, 32)`:
- `M_N` is symmetric.
- `eigvals` are finite and numerically stable.
- `rayleigh(x)` is finite for unit-norm `x`.
- `fiedler_vec` is orthogonal to the constant vector up to tolerance.
- Projecting to top-k eigvecs preserves batch dimension.

---

## 3. Phase 2 — `EntropicClock` (`esdm/clock.py`)

The clock replaces every external `t` integer and every fixed schedule in the system.

### 3.1 State variables

```python
@dataclass
class ClockState:
    tau_n: float          # accumulated entropic time (monotone)
    tau_max_est: float    # running estimate of total tau (for schedule normalisation)
    Lambda_tau: float     # entropic pump = 1 / |d(kl_bright)/d(phi)| + eps
    delta_tau: float      # |Delta kl_tau| at current step
    phi: float            # current clock field = log_lam_s
    dark_entropy: float   # KL of inactive modes (dark sector)
    conserved: float      # |kl_bright + dark_entropy - S_0|
    rho_sign: float       # rho_minus / (rho_plus + eps) — decoherence ratio
```

### 3.2 Module API

```python
class EntropicClock(nn.Module):
    """
    Stateful module accumulating entropic time from per-step kl_tau changes.
    All state lives in buffers (not parameters) so it is preserved across
    checkpoint save/load without contributing to the loss gradient.
    """

    def __init__(self, eps: float = 1e-8, heat_death_patience: int = 200): ...

    def step(
        self,
        kl_tau_current: Tensor,   # scalar or (F,) per-mode
        phi_current: Tensor,      # log_lam_s  — the clock field
        rho_plus: Tensor,         # constructive interference density
        rho_minus: Tensor,        # destructive interference density
        dark_entropy: Tensor,     # KL sum over inactive modes
        S_0: Tensor | None = None,
    ) -> ClockState: ...

    def alpha_bar(self, tau: Tensor | None = None) -> Tensor:
        """
        Entropic cosine schedule:
          alpha_bar(tau) = cos^2( pi/2 * tau / tau_max_est )
        If tau is None, uses self.tau_n (current accumulated time).
        """
        ...

    def is_heat_death(self) -> bool:
        """True if tau_n has been flat for > heat_death_patience steps."""
        ...
```

### 3.3 Lambda computation

```python
# discrete entropic pump (Barontini eq. 6 analogue)
d_kl_bright = kl_tau_current - self._prev_kl_tau       # Δ S_bright
d_phi       = phi_current    - self._prev_phi           # Δ phi
Lambda_tau  = 1.0 / (abs(d_kl_bright / (d_phi + eps)) + eps)
```

`Lambda_tau` is clipped to `[0, 100]` before use in the wave recurrence to prevent
explosion during the first few steps when `d_phi ≈ 0`.

### 3.4 Test checkpoint

`tests/test_clock.py`:
- `tau_n` is monotonically non-decreasing across 1000 random `kl_tau` sequences.
- `alpha_bar(0) ≈ 1.0`, `alpha_bar(tau_max) ≈ 0.0`.
- `is_heat_death()` triggers after the correct number of flat steps.
- `Lambda_tau` never exceeds clip threshold.

---

## 4. Phase 3 — `VibrationalStateBlock` modification (`esdm/vibrational.py`)

This is a **minimal-change** modification of the existing VDT `VibrationalStateBlock`.
Only the forward signature and the forcing line change.

### 4.1 Change diff (pseudocode)

```python
# BEFORE (VDT original)
Q_next = 2*Q - Q_prev - dt**2 * L_f @ Q - gamma * dQ + dt**2 * B_t

# AFTER (ESDM)
Q_next = 2*Q - Q_prev - delta_tau**2 * L_f @ Q - gamma * dQ + Lambda_tau * delta_tau**2 * B_t
```

where `delta_tau` and `Lambda_tau` come from `EntropicClock.step()`.

### 4.2 Interpretation

- `delta_tau` replaces `dt`: recurrence steps are now measured in entropic units.
  A step with large `|Δkl_tau|` takes a longer entropic stride.
- `Lambda_tau * B_t`: attention forcing is amplified when entropy is barely flowing
  (stable spectral geometry) and suppressed when modes are rapidly transitioning
  (direct analogue of Barontini's entropy-dependent pump).

### 4.3 Dual-space coupling in item-space

After the recurrence, optionally project `Q_t` through the fused item-space matrix:

```python
# dual-space filtered state in item-space (experimental)
Q_filtered = DualSpaceMatrix.project(Q_t.flatten(1), k=n_modes).reshape_as(Q_t)
```

This keeps the fusion exactly where training and inference happen: item-space.
Controlled by a boolean flag `use_dual_projection` (default `False`).

### 4.4 Test checkpoint

`tests/test_vibrational.py`:
- With `Lambda_tau=1.0` and `delta_tau=dt`, output matches original VDT numerically.
- With `Lambda_tau=0.0`, forcing term `B_t` vanishes from the recurrence.
- `Q_next` is finite and bounded for random inputs.

---

## 5. Phase 4 — `EntropicDiffusionDecoder` (`esdm/diffusion_decoder.py`)

Modifies the SDM reverse chain to use `EntropicClock.alpha_bar()` instead of
`cosine_alpha_bar(t, T=200)`.

### 5.1 Schedule replacement

```python
# BEFORE (SDM)
alpha_bar_t = cosine_alpha_bar(t, T=200)

# AFTER (ESDM)
alpha_bar_t = clock.alpha_bar(tau=clock.tau_n)

# per-mode variant (one tau_k per mode)
alpha_bar_k = clock.alpha_bar(tau=clock.per_mode_tau[k])
```

No other changes to the reverse chain, LGMRF prior, or noise schedule logic.

### 5.2 `per_mode_tau` tracking

`EntropicClock` accumulates a per-mode entropic time vector
`per_mode_tau ∈ R^F` when `kl_tau` is supplied as a vector:

```python
self.per_mode_tau += kl_tau_current.abs()  # element-wise accumulation
```

This enables different modes to reach their noise floor at different rates — a direct
analogue of Barontini's per-barrier-height $\tau$ curves (Fig. 2 of the paper).

### 5.3 Test checkpoint

`tests/test_diffusion_decoder.py`:
- `alpha_bar` is monotonically decreasing with `tau`.
- Per-mode `alpha_bar_k` is independent across modes.
- Full reverse-chain pass produces finite output with correct shape.

---

## 6. Phase 5 — ESDM ELBO (`esdm/elbo.py`)

### 6.1 Full ELBO

```python
def esdm_elbo(
    x: Tensor,             # input
    x_hat: Tensor,         # reconstruction
    kl_z: Tensor,          # KL(q(z) || N(0,I))
    kl_S: Tensor,          # KL(q(S) || p(S|I))   — coupling weight
    kl_tau: Tensor,        # KL(q(omega) || p(omega|tau,L))
    dark_entropy: Tensor,  # KL of inactive modes
    S_0: Tensor,           # initial total entropy (buffer, not parameter)
    beta: float = None,    # conservation weight; None → 1/S_0
) -> tuple[Tensor, dict]:
    """
    Returns (loss, metrics_dict).
    metrics_dict contains each term separately for logging.
    """
    recon   = -log_likelihood(x, x_hat)
    conserved = (kl_tau + dark_entropy - S_0).abs()
    if beta is None:
        beta = 1.0 / (S_0 + 1e-8)
    loss = recon + kl_z + kl_S + kl_tau + beta * conserved**2
    return loss, {
        "recon": recon, "kl_z": kl_z, "kl_S": kl_S,
        "kl_tau": kl_tau, "dark_entropy": dark_entropy,
        "conserved": conserved, "beta": beta,
    }
```

### 6.2 `S_0` initialisation

`S_0` is computed from the **first forward pass only** and stored as a non-gradient buffer
in the top-level `ESDM` module. It must not change after the first step.

### 6.3 Test checkpoint

`tests/test_elbo.py`:
- All five terms are non-negative.
- `conserved → 0` when `kl_tau + dark_entropy == S_0` exactly.
- Gradient flows to `kl_z`, `kl_S`, `kl_tau` but not to `S_0`.
- `S_0` is set once and remains unchanged across later batches.

---

## 7. Phase 6 — Top-Level `ESDM` Module (`esdm/model.py`)

### 7.1 Forward pass

```python
class ESDM(nn.Module):

    def forward(self, x: Tensor) -> ESDMOutput:
        # 1. Spectral loading: produce log_lam_s, N_active, L_z, kl_tau, kl_S
        spectral = self.spectral_decoder(x)

        # 2. First-batch S_0 initialisation
        if self.S_0 is None:
            self.S_0 = (spectral.kl_tau + spectral.dark_entropy).detach()

        # 3. Entropic clock: update tau, Lambda, delta_tau
        clock_state = self.clock.step(
            kl_tau_current = spectral.kl_tau,
            phi_current    = spectral.log_lam_s,
            rho_plus       = spectral.rho_plus,
            rho_minus      = spectral.rho_minus,
            dark_entropy   = spectral.dark_entropy,
            S_0            = self.S_0,
        )

        # 4. Wave recurrence: Q_t gated by Lambda(tau)
        Q, rho_plus, rho_minus = self.vibrational(
            x, clock_state.Lambda_tau, clock_state.delta_tau
        )

        # 5. Optional item-space projection through M_N
        if self.use_dual_projection:
            Q = self.dual_space.project(Q.flatten(1), k=self.n_modes).reshape_as(Q)

        # 6. Entropic diffusion decode: alpha_bar from clock
        x_hat = self.diffusion_decoder(Q, spectral.L_z, self.clock)

        # 7. ELBO
        loss, metrics = esdm_elbo(
            x, x_hat,
            kl_z         = spectral.kl_z,
            kl_S         = spectral.kl_S,
            kl_tau       = spectral.kl_tau,
            dark_entropy = clock_state.dark_entropy,
            S_0          = self.S_0,
        )

        return ESDMOutput(x_hat=x_hat, loss=loss, clock=clock_state, metrics=metrics)
```

### 7.2 Health monitoring hook

```python
    def health_check(self) -> dict:
        """
        Returns the diagnostic table from 00-DESIGN.md §8.1.
        Intended to be called every N batches, not every step.
        """
        c = self.clock
        return {
            "tau_n":       c.tau_n,
            "tau_flat":    c._flat_steps,
            "conserved":   c.conserved,
            "rho_sign":    c.rho_sign,
            "N_active":    self._last_N_active,
            "Lambda_tau":  c.Lambda_tau,
            "heat_death":  c.is_heat_death(),
        }
```

---

## 8. Phase 7 — Item-Space Fusion Experiments

These are validation experiments, not training runs.

### 8.1 Experiment E1 — Fused item kernel validation

Compare:
- cosine similarity from `VV^T`
- pure semantic kernel from `VL_FV^T`
- fused kernel `M_N`

Metrics:
- Eigenvalue decay rate
- `rayleigh(v)` distribution under each kernel
- Jaccard overlap between top-k neighbours under `M_N` vs. cosine vs. SPIN lambda-aware

Expected result: `M_N` separates geometric and semantic outliers more cleanly than
pure geometry alone, while remaining fully usable in item-space training and inference.

### 8.2 Experiment E2 — Fiedler vector tracking

During training, log `fiedler_vec` of `M_N` every 50 steps. Plot cosine angle between
the Fiedler vector and the bottom eigenvector of the amplitude Laplacian from the
vibrational block. Per the Spectral Complementarity Conjecture, this angle should
tend toward 90° as training stabilises.

### 8.3 Experiment E3 — Barrier height sweep

Following 00-DESIGN.md §9, sweep `kl_S ∈ {0.001, 0.01, 0.05, 0.1, 0.5}` and plot
`tau_n` vs. training step for each value. Reproduce Barontini Fig. 2 in semantic space.

---

## 9. Implementation Sequence

| Phase | File | Depends on | Estimated complexity | Status |
|-------|------|------------|----------------------|--------|
| 1 | `laplacian.py` | ArrowSpace GW output | Medium | ✅ |
| 2 | `clock.py` | — | Low | ✅ |
| 3 | `vibrational.py` | `clock.py` | Low (minimal diff) | ✅ |
| 4 | `diffusion_decoder.py` | `clock.py` | Low (schedule swap) | ✅ |
| 5 | `elbo.py` | `clock.py` | Low | ✅ |
| 6 | `model.py` | All of above | Medium | ✅ (smoke-test wiring) |
| 7 | Experiments | `model.py`, `laplacian.py` | Medium–High | ⬜ |

Start with Phase 1 and 2 in parallel (they have no mutual dependency).
Do not begin Phase 6 until Phases 1–5 each pass their unit tests independently.

---

## 10. Dependency Pinning

```text
torch>=2.3.0
numpy>=1.26
scipy>=1.13          # sparse eigensolver for L_F and M_N
einops>=0.8          # for vibrational block reshaping
pytest>=8.0
```

ArrowSpace `L_F` is expected as a pre-computed `torch.Tensor` loaded from a `.pt` file
produced by `scripts/build_laplacian.py`. The Rust arrowspace library is not called at
training time.
