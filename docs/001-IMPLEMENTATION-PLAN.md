# 001 — ESDM Implementation Plan

> **Status:** Implementation-active (rewritten 2026-07-15, issue #20)
> **Depends on:** `docs/00-DESIGN.md`, `docs/00-DESIGN_HYBRID.md`
> **Target:** PyTorch >= 2.3, Python >= 3.12

## 1. Scope

This document defines the concrete implementation contract for the ESDM
entropic-clock track: the two-sector partition, the entropic clock, the
hybridisation operator $\mathcal{H}$, the ELBO, and the fitness criteria.

The 2.5-D dual-space track (score3d/score25d) is defined separately in
`002-EXPERIMENTS.md` and is out of scope here.

## 2. Design principles (non-negotiable)

These are constants derived from the design docs, not hyperparameters.
They must not be swept, parameterised, or overridden without a design-doc
amendment.

| Constant | Value | Source |
|---|---|---|
| $q$ (spectral modes) | $F/4$ (rounded to nearest power of 2) | `00-DESIGN_HYBRID.md` §5 |
| $\gamma$ (damping) | 0.1 | VDT default |
| `n_vibrational_steps` | 2 | CFL-safe for real $L_F$ |
| `lambda_clip` | derived: $0.1 F$ | `00-DESIGN.md` §4.2 (Λ/H balance derivation) |
| `grad_clip` | 1.0 | Training stability |
| `seed` | 3407 | Project-wide |
| `batch_size` | 64 | Standard |
| `k` (ArrowSpace clusters) | 4 | Graph construction |
| `mode_threshold` | median of $\omega$ at init | §3.1 below |
| `tau_max_est` | $S_0 \cdot 3$ | §4.1 below |
| $\sigma^2$ (diffusion coefficient) | 1.0 | `00-DESIGN_HYBRID.md` §5 |
| $\eta$ (Rayleigh step size) | 10.0 | Sweep-derived constant, not a free parameter |

The only free hyperparameters are `lr` and `n_steps` (governed by heat death).

## 3. Two-sector partition

### 3.1 Mode threshold

The design (`00-DESIGN.md` §2.2) says: bright sector = modes where
$\omega_k > \theta$, dark sector = inactive modes. The threshold $\theta$
must be calibrated to the omega scale, not a magic number.

At initialisation, `omega_net` has near-zero weights and zero bias, so
$\omega_k = \exp(0) \approx 1.0$ for all modes. A fixed threshold of 0.01
puts ALL modes in the bright sector, making `dark_entropy = 0` and breaking
the two-sector partition.

**Fix:** the mode threshold is the **median** of $\omega$ across modes,
computed per-batch at each step. This ensures ~half the modes are bright
and ~half are dark, matching the Barontini two-sector analogue (equal
partition). The threshold is not a parameter — it is a derived quantity.

### 3.2 Dark entropy

`dark_entropy` is the KL sum over inactive modes ($\omega_k \leq \theta$).
With the median threshold, this is always non-zero and the conservation
term $\beta \cdot |S_\mathit{bright} + S_\mathit{dark} - S_0|^2$ has a real
signal to work with.

## 4. Entropic clock

### 4.1 tau_max_est

`tau_max_est` is the entropic time horizon for the cosine schedule
$\bar{\alpha}(\tau) = \cos^2(\pi/2 \cdot \tau / \tau_\mathit{max})$.

The clock accumulates raw $|\Delta\text{kl}_\tau|$ (not normalised by $S_0$).
On CIFAR-10 with $S_0 \approx 52$, per-step deltas are $\sim 0.2$ nats,
so a full 800-step run accumulates $\tau_n \approx 160$.

**Fix:** `tau_max_est` is set to $3 \cdot S_0$ at initialisation. With
$S_0 \approx 52$, this gives $\tau_\mathit{max} \approx 156$, spreading the
cosine schedule across the full run. This is a derived constant, not a
parameter.

### 4.2 Heat death

Heat death fires when $\tau_n$ stalls for > `heat_death_patience` steps.
The relative threshold `eps_rel = 1e-3` catches convergence on real
stochastic batches. If the model is learning, KL stabilises, $\Delta\tau$
drops, and heat death fires naturally at ~70% through the run.

## 5. Hybridisation operator $\mathcal{H}$

Defined in `00-DESIGN_HYBRID.md`. Three steps, applied in sequence in
`VibrationalStateBlock._step()`:

$$\mathcal{H}(Q_t) = \Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)$$

### 5.1 Sign convention

The design doc defines $\Pi = -\eta \nabla_R$ and the recurrence as
$-\Delta\tau^2 \Pi$, which gives $+\eta \Delta\tau^2 \nabla_R$
(destabilising — pushes toward high energy).

The correct wave-equation convention: $\mathcal{H}_\mathit{action} = \eta \nabla_R$
(analogous to $L Q$), so $-\Delta\tau^2 \mathcal{H}_\mathit{action}$ is the
restoring force toward low-energy modes. This is the implemented sign.

### 5.2 Parameters

$\sigma^2 = 1.0$ and $\eta = 10.0$ are constants (not swept). $\eta = 10.0$
was identified by parameter sweep as the value where
`pi_contraction_ratio` enters the healthy $[0.01, 0.3]$ range. Both are
learnable `nn.Parameter` objects so the optimizer can fine-tune them
during training, but their initial values are fixed constants.

### 5.3 Diagnostics

Two diagnostics must be logged per step:

- `net_energy_flow` $= \|Q_{t+1}\|_F^2 - \|Q_t\|_F^2$ — $\Lambda$ vs $\mathcal{H}$ balance
- `pi_contraction_ratio` $= \|\Delta Q_t\| / \|Q_t\|$ — $\Pi$ activity

Healthy ranges per `00-DESIGN_HYBRID.md` §7:
- `net_energy_flow` $\approx 0$ (balanced)
- `pi_contraction_ratio` $\in [0.01, 0.3]$

## 6. ELBO and reconstruction

### 6.1 ELBO

$$\mathcal{L} = \text{recon} + \text{kl}_z + \text{kl}_S + \text{kl}_\tau + \beta \cdot |S_\mathit{bright} + S_\mathit{dark} - S_0|^2$$

where $\beta = 1/S_0$.

### 6.2 Decoder

`EntropicDiffusionDecoder` computes:
$$\hat{x} = \sqrt{\bar{\alpha}(\tau)} \cdot Q + \sqrt{1 - \bar{\alpha}(\tau)} \cdot \epsilon$$

With $\epsilon = 0$ (deterministic reconstruction), $\hat{x} = \sqrt{\bar{\alpha}} \cdot Q$.
At init, $\bar{\alpha} \approx 1$ so $\hat{x} \approx Q \approx x$. As $\tau$
accumulates, $\bar{\alpha} \to 0$ and $\hat{x} \to 0$.

**Decoder collapse guard:** if `recon < 1e-3` at any point, the model has
found the trivial constant-output optimum. The fitness function scores
this as 0.

### 6.3 Loss tracking

The total ELBO can increase during training because KL terms grow as the
encoder learns — this is normal VAE behaviour. The fitness function must
track **reconstruction loss** (`recon`), not total ELBO (`loss`), for the
`loss_decrease` criterion.

## 7. Fitness criteria

Defined in `esdm/eval/fitness.py`, computed from exp4/exp5 metrics.
Maximum score: 85.

| Criterion | Max | What it measures |
|---|---|---|
| `tau_monotone` | 10 | $\tau_n$ strictly increasing |
| `tau_range_bonus` | 5 | Total $\tau_n$ accumulation |
| `alpha_bar_decrease` | 10 | $\bar{\alpha}(\tau)$ drops from ~1 toward ~0 |
| `conservation` | 15 | $\|S_\mathit{total} - S_0\| / S_0 \to 0$ |
| `heat_death` | 10 | Heat death fires within the run |
| `heat_death_timing` | 5 | Fires around 70% through |
| `rho_sign_stabilizing` | 10 | $\rho_-/\rho_+$ variance decreasing |
| `recon_decrease` | 10 | Reconstruction loss decreases (NOT total ELBO) |
| `net_energy_flow` | 5 | $\Lambda$ vs $\mathcal{H}$ balance ($\approx 0$) |
| `pi_contraction_ratio` | 5 | $\Pi$ activity $\in [0.01, 0.3]$ |

## 8. ArrowSpace usage contract

All graph construction and spectral computation goes through
`esdm/graphs/wiring.py` (the `WireGraph` adapter), which wraps the
`arrowspace` Rust bindings. No custom Laplacian code outside the adapter.

## 9. Validation requirements (issue #27)

Before any architecture claim is published:

1. **Frozen-gate ablation:** frozen $\mathcal{H}$ (fixed $\eta$, $\sigma^2$) vs learned. Delta must be reported.
2. **Step-count sweep:** $N \in \{1, 2, 4, 8, 16, 32\}$ reverse steps.
3. **Diversity/quality split:** coverage + precision alongside fitness.
4. **Non-spectral baseline:** plain MLP at matched parameter count.
5. **Conservation reporting:** terminal-step `conserved` residual in all results tables.
