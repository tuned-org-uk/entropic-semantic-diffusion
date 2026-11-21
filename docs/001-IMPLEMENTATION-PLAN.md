# 001 — ESDM Implementation Plan

> **Status:** v2.0.0 — the [v2 reconstruction contract](00-DESIGN.md#v2-reconstruction-contract)
> supersedes historical sphere and decoder requirements in this plan for the
> entropic-clock track. See [parallel work items](v2-work-items.md).
> **Depends on:** `docs/00-DESIGN.md`
> **Target:** PyTorch >= 2.3, Python >= 3.12

## 1. Scope

This document defines the concrete implementation contract for the ESDM:
the two-sector partition, the entropic clock, the semantic operator
$\mathcal{H}$, the $S^{F-1}$ geometric constraint, the topology-adaptive
decoder, and the fitness criteria.

## 2. Design principles (non-negotiable)

These are constants derived from the design docs, not hyperparameters.

| Constant | Value | Source |
|---|---|---|
| $q$ (spectral modes) | $F/4$ (rounded to nearest power of 2) | `00-DESIGN.md` §4.2 |
| $\gamma$ (damping) | 0.1 | VDT default |
| `n_vibrational_steps` | 2 | CFL-safe for real $L_F$ |
| $\lambda_\mathit{clip}$ | derived: $0.1\,F$ | `00-DESIGN.md` §4.6 |
| `grad_clip` | 1.0 | Training stability |
| `seed` | 3407 | Project-wide |
| `batch_size` | 64 | Standard |
| $k$ (ArrowSpace clusters) | 4 | Graph construction |
| `mode_threshold` | median of $\omega$ per-batch | §3.1 |
| `tau_max_est` | $3 \cdot S_0$ (derived) | §4.1 |
| $\sigma^2$ (diffusion coefficient) | 1.0 | `00-DESIGN.md` §4.3 |
| $\eta$ (Rayleigh step size) | derived: $0.1\,F / (2\lambda_\mathit{max})$ | `00-DESIGN.md` §4.4 |
| `n_steps` (CIFAR-10) | 700 | `dataset_cifar10.yaml` |

The only free hyperparameters are `lr` and `n_steps` (governed by heat death).

## 3. Two-sector partition

### 3.1 Mode threshold

The threshold $\theta$ is the **median** of $\omega$ across modes, computed
per-batch at each step. This ensures ~half the modes are bright and ~half
are dark. The threshold is not a parameter — it is a derived quantity.

### 3.2 Dark entropy

`dark_entropy` is the KL sum over inactive modes ($\omega_k \leq \theta$).
With the median threshold, this is always non-zero and the conservation
term has a real signal to work with.

## 4. Entropic clock

### 4.1 tau_max_est

`tau_max_est` is set to $3 \cdot S_0$ at initialisation. With
$S_0 \approx 52$ on CIFAR-10, this gives $\tau_\mathit{max} \approx 156$.

The clock accumulates raw $|\Delta\mathit{kl\_tau}|$ (not normalised by $S_0$).

### 4.2 Heat death

Heat death fires when $\tau_n$ stalls for > `heat_death_patience` steps.
The relative threshold `eps_rel = 1e-3` (CIFAR-10) catches convergence on
real stochastic batches.

### 4.3 Lambda gating

The entropic pump $\Lambda(\tau)$ is gated by $\alpha_\mathit{bar}(\tau)$:
$\Lambda_\mathit{eff} = \Lambda \cdot \alpha_\mathit{bar}$. This couples the
pump to the diffusion schedule: full strength early (exploration), ramping
to zero late (convergence, letting $\mathcal{H}$ pull $Q$ toward smooth modes
without pump interference).

## 5. Semantic operator $\mathcal{H}$

Defined in `00-DESIGN.md` §4. Three steps, applied in sequence in
`VibrationalStateBlock._step()`:

$$\mathcal{H}(Q_t) = \Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)$$

### 5.1 Sign convention

$\mathcal{H}_\mathit{action} = \eta \nabla_R$ (analogous to $L Q$), so
$-\Delta\tau^2 \mathcal{H}_\mathit{action}$ is the restoring force toward
low-energy modes. This is the implemented sign.

### 5.2 Parameters

$\sigma^2 = 1.0$ and $\eta$ are derived constants (not swept). $\eta$ is
derived at init as $0.1\,F / (2\lambda_\mathit{max}(L_F))$ from the spectral
scale of $L_F$. Both are learnable `nn.Parameter` objects.

### 5.3 $S^{F-1}$ normalisation

After each wave recurrence step, $Q$ is projected back onto $S^{F-1}$ via
L2 normalisation. This is the geometric constraint that prevents CFL
instability: without it, $Q$ drifts off the manifold and the Rayleigh
gradient in $\mathcal{H}$ shrinks (normalised by $\|\tilde{F}_t\|^2$),
causing the wave recurrence to become an energy amplifier.

### 5.4 Diagnostics

Three diagnostics are logged per step:

- `angular_displacement` $= \arccos(Q_{t+1} \cdot Q_t)$ — $Q$ movement on
  $S^{F-1}$. Replaces `net_energy_flow` (which was identically 0 with
  normalisation). Healthy range $[0.01, 0.5]$ radians.
- `pi_contraction_ratio` $= \|\Delta Q_t\| / \|Q_t\|$ — $\Pi$ activity.
  With $\|Q\|=1$, directly measures Rayleigh gradient magnitude. Healthy
  range $[0.3, 3.5]$ (IQR-calibrated for $S^{F-1}$).
- `gamma_mean` — mean damping coefficient. Monitored to detect gamma collapse.

## 6. ELBO and reconstruction

### 6.1 ELBO

$$\mathcal{L} = \text{recon} + \text{kl}_z + \text{kl}_S + \text{kl}_\tau + \beta \cdot |S_\mathit{bright} + S_\mathit{dark} - S_0|^2$$

where $\beta = 1/S_0$.

### 6.2 Decoder

`EntropicDiffusionDecoder` computes:
$$\hat{x} = \sqrt{\bar{\alpha}(\tau)} \cdot Q + \sqrt{1 - \bar{\alpha}(\tau)} \cdot \epsilon$$

With $\epsilon = 0$: $\hat{x} = \sqrt{\bar{\alpha}} \cdot Q$. At init,
$\bar{\alpha} \approx 1$ so $\hat{x} \approx Q \approx x$. As $\tau$
accumulates, $\bar{\alpha} \to 0$ and $\hat{x} \to 0$.

### 6.3 Topology-adaptive decoder

The `TopologyDecoder` (`esdm/models/topology_decoder.py`) is the
generation-side decoder. It projects $Q$ onto the $K$ smooth eigenvectors
($U_q$), decodes from the $K$-dim spectral chart, and lifts back to ambient
space. The heat kernel prior $\mathcal{N}(0, e^{-2t L_K})$ on the spectral
chart ensures samples lie on the smooth submanifold by construction.

See `00-DESIGN.md` §1.4 and §12 for the full design.

## 7. Fitness criteria

Defined in `esdm/eval/fitness.py`, computed from exp5 metrics.
Maximum score: 85.

| Criterion | Max | What it measures |
|---|---|---|
| `tau_monotone` | 10 | $\tau_n$ strictly increasing |
| `tau_range_bonus` | 5 | Total $\tau_n$ accumulation |
| `alpha_bar_decrease` | 10 | $\bar{\alpha}(\tau)$ drops from ~1 toward ~0 |
| `conservation` | 15 | $\|S_\mathit{total} - S_0\| / S_0 \to 0$ (tautological — enforced by construction) |
| `heat_death` | 10 | Heat death fires within the run |
| `heat_death_timing` | 5 | Fires around 70% through |
| `rho_sign_stabilizing` | 10 | $\rho_-/\rho_+$ variance decreasing |
| `recon_decrease` | 10 | Median recon during active phase ($\alpha_\mathit{bar} > 0.01$) |
| `angular_displacement` | 5 | $\arccos(Q_{t+1} \cdot Q_t)$ — $Q$ movement on $S^{F-1}$ |
| `pi_contraction_ratio` | 5 | $\Pi$ activity $\in [0.3, 3.5]$ |

Current baseline: **84.48/85 (99.4%)** on CIFAR-10 DINO-256, 700 steps, lr=0.025.

## 8. ArrowSpace usage contract

All graph construction and spectral computation goes through
`esdm/graphs/wiring.py` (the `WireGraph` adapter), which wraps the
`arrowspace` Rust bindings. `WireGraph` stores both the graph Laplacian
(`laplacian_dense`) and the ArrowSpace lambdas (`lambdas` field) — the
full ArrowSpace structure, not just the matrix.

## 9. Samplers

Three samplers exist, in order of architectural sophistication:

1. **DDPM sampler** (`entropic_sampler.py`): operates in input space.
   Architecturally mismatched with ESDM (diffusion is internal to $Q$).
   Kept for backward compatibility. Produces zero embeddings on real data.

2. **Q-space sampler** (`q_space_sampler.py`): score-approximation in $Q$-space.
   U_q-constrained initialisation, DDIM-style reverse step, model forward
   pass as score predictor. Produces non-zero embeddings with correct
   marginal stats but off-manifold (NN ratio ~74).

3. **Topology sampler** (`topology_decoder.py`): ArrowSpace-aware reverse
   diffusion in spectral coordinates. Heat kernel prior on the projected
   Laplacian $L_K = U_q^\top L_F U_q$. Sampling in spectral coords is
   on-manifold by construction. Currently still off-manifold (NN ratio ~81)
   — the heat kernel prior captures the graph geometry but not yet the
   data distribution (lambda weighting is implemented but not yet
   integrated into the sampler loop).
