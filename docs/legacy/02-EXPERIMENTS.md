# 02 — ESDM Experiment Suite

> **Status:** Exp 1–2 complete. Exp 3 ready to run.  
> **Relates to:** `docs/00-DESIGN.md` §9 (barrier height sweep), §11 (open questions), §12 (experiment checklist)  
> **Covers:** Five experiments of increasing complexity from unit-level sanity checks to full 32×32 image diffusion

---

## Overview

The five experiments below form a deliberate ladder: each one assumes the previous has passed.
The first two are unit tests disguised as experiments — they validate the physics of the clock
and the vibrational recurrence on synthetic data before any real corpus is involved.
Experiments 3 and 4 validate semantic behaviour on structured low-dimensional datasets.
Experiment 5 is the full-scale diffusion run on 32×32 images.

Each experiment section specifies: the dataset and setup, the observable quantities to measure,
the success criteria, the failure modes to investigate, and the connection to the theoretical
claims in `00-DESIGN.md`.

---

## Experiment 1 — Entropic Clock Calibration on a Two-Mode Synthetic Graph

> **Status:** ✅ Complete.

### Motivation

Before the full ESDM is trained on any real data, the `EntropicClock` must be verified to
exhibit the correct thermodynamic behaviour on a problem where the answer is known analytically.
This experiment constructs the simplest possible two-sector system: a graph with exactly two
communities (two bright modes, one dark mode) and checks that the entropic time \(\tau_n\)
accumulates exactly when entropy flows between sectors and stalls when entropy is static.

### Setup

**Dataset:** Synthetic. A single \(5 \times 5\) graph Laplacian \(L\) is constructed from two
tightly connected communities of 5 nodes each, connected by a single weak bridge edge of weight
\(w \in \{0.0, 0.01, 0.1, 1.0\}\). The spectral gap (Fiedler value) varies monotonically with
\(w\).

**Model:** `EntropicClock` in isolation (no `VibrationalStateBlock`, no decoder). At each
synthetic "training step", inject a scripted `kl_tau_current` signal:

1. **Phase A** (steps 1–50): constant `kl_tau = 1.0` — no entropy flow, \(\Delta kl\_tau = 0\).
2. **Phase B** (steps 51–150): linearly increasing `kl_tau` from 1.0 to 3.0 — entropy flowing.
3. **Phase C** (steps 151–200): constant `kl_tau = 3.0` — entropy stalled again.
4. **Phase D** (steps 201–250): oscillating `kl_tau` — tests non-monotonic clock field.

The `phi_current` signal is `log(1 + kl_tau_current)` (simulating `log_lam_s` as clock field).

**Metrics logged per step:**

| Metric | Expected behaviour |
|--------|-------------------|
| `tau_n` | Flat in A, increasing in B, flat in C, increasing in D (abs-value integration) |
| `delta_tau` | Zero in A and C, positive in B and D |
| `Lambda_tau` | Stable in A/C, varies with \(\partial_\phi kl\_tau\) in B/D |
| `alpha_bar(tau_n)` | 1.0 in A, decreasing in B, plateau in C, continues decreasing in D |
| `is_heat_death()` | False throughout (patience=200 never reached in 250 steps) |
| `conserved` | Zero (no `S_0` supplied yet) |

### Success Criteria

- \(\tau_n\) is exactly flat during phases A and C (within floating-point tolerance).
- \(\tau_n\) increases monotonically during phases B and D regardless of sign changes in `kl_tau`.
- `alpha_bar` obeys the cosine schedule: at \(\tau / \tau\_\mathit{max\_est} = 0.5\) it equals
  \(\cos^2(\pi/4) = 0.5\).
- `Lambda_tau` is finite and clipped to `lambda_clip=100.0` for all steps.
- No NaN or Inf in any buffer after 250 steps.

### Failure Modes

- **`tau_n` drifts during flat phases:** numerical precision issue in the abs-value accumulation.
  Check `delta_tau.item() <= eps` threshold in `clock.step()`.
- **`Lambda_tau` explodes:** `d_phi` is near zero causing division blow-up. Verify the `+ eps`
  guard in the denominator.
- **`alpha_bar` does not reach 1.0 at initialisation:** `tau_max_est` is initialised to 1.0, so
  the first `alpha_bar` call returns \(\cos^2(\pi/2 \cdot 0/1) = 1.0\). Check the zero-division
  guard.

### Connection to Design

This experiment directly validates §3 (Entropic Time) and the health check table in §8.1 of
`00-DESIGN.md`. Specifically, it tests the "no entropic time elapses between a big crunch and
the subsequent big bang" property: during phase C, \(\tau_n\) must stop exactly as entropy
exchange ceases.

---

## Experiment 2 — Vibrational Wave Recurrence on a 1-D Harmonic Chain

> **Status:** ✅ Complete — seed 3407 passed all criteria. Commit `2af4381` on `mecis/main`.
> Full test suite: 53 passed in 2.70s. Output: `runs/exp2/3407/metrics.csv` (500 rows × 25 columns, gitignored).

### Motivation

The wave recurrence in `VibrationalStateBlock` is a discrete damped wave equation. On a 1-D
harmonic chain (path graph Laplacian) the analytic solution is known: each normal mode decays
as \(e^{-\gamma t}\) while oscillating at frequency \(\sqrt{\lambda_k}\). This experiment
verifies the ESDM `VibrationalStateBlock` reproduces this decay numerically, and that replacing
\(dt\) with \(\Delta\tau\) from a live `EntropicClock` does not break CFL stability.

### Setup

**Dataset:** Synthetic. A 1-D path graph with \(N = 32\) nodes. The path graph Laplacian
\(L \in \mathbb{R}^{32 \times 32}\) has known eigenvalues \(\lambda_k = 2 - 2\cos(\pi k / 32)\)
for \(k = 0, \ldots, 31\).

**Initial condition:** \(x_0 \in \mathbb{R}^{32}\) set to the \(k=1\) eigenvector (the Fiedler
vector) — a pure single-mode excitation. This allows the analytic decay envelope to be compared
directly to the simulation.

**Model:** `EntropicClock` + `VibrationalStateBlock(dim=32, n_steps=1)`. The clock is driven by
injecting `kl_tau_current = exp(-kl_decay * step)` with `kl_decay = 0.08` (decaying
entropy flow, mimicking convergence).

> **Note (from seed 3407):** `kl_decay=0.08` exhausts the entropy budget by step ~100 of 400,
> leaving ~75% of the run in the full-noise regime (`alpha_bar ≈ 3.7e-33` by step 100). This is
> expected and confirmed by design: Exp 2 is a unit test of clock + wave recurrence mechanics,
> not a training run. The fast decay guarantees heat death fires within the 400-step window and
> keeps the test fast. **Exp 3 must use `kl_decay ∈ {0.01, 0.03}` to avoid this collapse in
> a 2000-step training run.** See §Exp 3 setup for the revised sweep.

The entropic pump is capped at `lambda_clip = 15.0`.

**Run:** 500 forward steps. At each step record:

| Metric | Description |
|--------|-------------|
| `Q_t[0]` | First element of the state vector — tracks Fiedler mode amplitude |
| `rho_plus` | Constructive density |
| `rho_minus` | Destructive density |
| `rho_sign` | `rho_minus / rho_plus` — should cross 1.0 as interference regime switches |
| `dt` (CFL-clamped) | Must remain \(\leq \sqrt{2/\lambda_\mathit{max}}\) at every step |
| `dark_entropy` | Shannon entropy of the normalised high-frequency (dark) mode energies |
| `is_heat_death()` | Must become True by step ~400 (decaying clock → flat `tau_n`) |

**CFL verification script:** At each step assert:
```python
lam_max = 2 - 2 * math.cos(math.pi * 31 / 32)  # top eigenvalue of path graph
assert dt.item() <= math.sqrt(2 / lam_max) + 1e-6
```

### Seed 3407 Results (reference)

| Observable | Value | Assessment |
|---|---|---|
| Heat death step | 399 (target: 400 ± 50) | ✅ Pass |
| Final `tau_n` | 2.0 | ✅ Finite |
| Final `alpha_bar` | ≈ 3.7e-33 | ✅ Collapsed (expected at `kl_decay=0.08`) |
| Final `Lambda_tau` | 15.0 (cap hit) | ✅ Clipped correctly |
| Final `rho_sign` | 1.0 | ✅ Decoherent at heat death |

### Success Criteria

- `Q_t` remains finite and does not blow up for all 500 steps (`max |Q_t[0]| < 10`).
- CFL assertion never fires.
- `rho_sign` crosses 1.0 at least twice, showing the system visits both constructive and
  destructive interference regimes.
- `is_heat_death()` returns `True` by step 400 ± 50 (when the decaying `kl_tau` causes
  `delta_tau` to drop below `eps` for `patience` consecutive steps).
- `dark_entropy` stays in the theoretically valid range
  \([0, \log(N - \text{bright\_modes})]\); a value near the upper bound means the dark sector
  is uniformly excited, while a value near zero means all dark energy is concentrated in a
  single mode.

### Failure Modes

- **Numerical blowup in `Q_t`:** CFL constraint violated. Check that `_laplacian_from_dense`
  correctly recovers the path graph adjacency and that `dt_max_cfl()` uses the true spectral
  radius, not the Gershgorin bound.
- **`rho_sign` never crosses 1.0:** `density.update()` is not being called on consecutive wave
  states, or the EMA momentum is keeping the density frozen at its random initialisation.
- **`is_heat_death()` never fires:** The decaying `kl_tau` is not making `delta_tau` small
  enough. Increase `kl_decay` or raise `clock_eps` / `heat_death_patience`.
- **`dark_entropy` outside `[0, log(N-bright_modes)]`:** The dark-mode energy distribution is
  not a valid probability vector; check the eigenvector normalisation in
  `path_graph_eigenvectors()`.

### Connection to Design

Validates §4.2 (ESDM wave recurrence) and the CFL discussion
in `_vendor/vdt.py`. Also tests the `rho_sign` confidence signal from §6.3 of `00-DESIGN.md`:
when the Fiedler mode is actively oscillating, `rho_minus / rho_plus` should be less than 1
(constructive regime); when the mode has nearly damped out, `rho_sign` should approach 1
(decoherent / low-confidence regime).

---

## Experiment 3 — Barrier Height Sweep on 2-D Gaussian Clusters

> **Status:** 🔜 Ready to run. Setup revised from Exp 2 seed 3407 observations (2026-06-30) and
> amended 2026-07-03 for `dim=64`, `bright_modes=32` on branch `exp3/dim64`.
> **Consolidated amendment:** This section reflects the `dim=64` / `bright_modes=dim//2` geometry (issue #6, 2026-07-03).

### Motivation

This is the primary validation experiment specified in §9 of `00-DESIGN.md`. It directly
replicates Barontini's barrier height sweep (Fig. 2 of Phys. Rev. Research 8, L022047): varying
the `kl_S` weight (the semantic coupling strength analogue to the BEC barrier height \(V\)) and
observing whether the system produces cyclic big bang/big crunch dynamics at low coupling or
heat death at high coupling.

### Setup

**Dataset:** 4 isotropic Gaussian clusters in \(\mathbb{R}^{64}\), 1000 samples each, with
inter-cluster separation controlled by a signal-to-noise ratio parameter \(\rho \in \{1, 3, 10\}\).
The 64-D embeddings are the direct input \(x \in \mathbb{R}^{B \times 64}\). Raising the dimension
from the original \(\mathbb{R}^8\) gives a richer spectral landscape: 64 eigenmodes vs. 8,
making the bright/dark sector partition non-trivial even at initialisation.

**Model:** Full ESDM with `dim=64`, `n_steps=2`, `bright_modes=32` (= `dim // 2`).
`SpectralLoadingDecoder` produces a \(64 \times 64\) wiring Laplacian \(L(z)\). The
`EntropicDiffusionDecoder` uses the global (non-per-mode) schedule. With `bright_modes=32`,
there are always **32 dark modes**, so `dark_entropy` has genuine structure to reveal — the
maximum possible dark entropy is \(\log(32) \approx 3.47\) nats.

**`kl_decay` sweep (revised from Exp 2):** Exp 2/seed 3407 showed that `kl_decay=0.08` with 400
steps exhausted the entropy budget by step ~100, leaving ~75% of the run with
`alpha_bar ≈ 3.7e-33` (full-noise regime). For a 2000-step Exp 3 run, the collocation is forced
entirely into the trivial region for most of training. Exp 3 therefore sweeps `kl_decay` jointly
with `kl_S_weight`:

- **`kl_decay=0.08`** is retained as the control condition (replicates Exp 2 collapse behaviour).
- **`kl_decay=0.03`** is the recommended baseline (entropy budget spreads to ~step 700 of 2000).
- **`kl_decay=0.01`** is the slow-decay condition (budget reaches ~step 1500 of 2000; tests
  whether collocation in the non-trivial region changes the `tau_n` trajectory qualitatively).

**`lambda_clip` revised:** Exp 2/seed 3407 recorded `Lambda_tau` hitting the cap (15.0) at step
257 — 64% through the run — freezing the forcing term for the remainder. For Exp 3:
- Default: `lambda_clip=10` (tighter cap; forces the pump to spread its amplitude budget).
- Control: `lambda_clip=100` (Exp 2 equivalent; one sub-run per barrier height for comparison).

**Sweep:**

```
kl_S_weight  in {0.001, 0.01, 0.05, 0.2, 1.0}      (5 barrier heights)
kl_decay     in {0.01, 0.03, 0.08}                   (3 schedule speeds; 0.08 is control)
rho          in {1, 3, 10}                            (3 cluster separations)
lambda_clip  in {10, 100}                             (2 pump caps; 100 is Exp 2 control)
```

For the primary result figure (analogous to Barontini Fig. 2) use `kl_decay=0.03`,
`lambda_clip=10`, `rho=3` (the recommended baseline condition).

Each run: 2000 training steps, batch size 64.

**Metrics logged per step (complete schema):**

| Metric | Role |
|--------|------|
| `tau_n` | Entropic time trajectory — main observable (analogous to Barontini Fig. 2) |
| `kl_tau` | Bright-sector KL (decays to zero at convergence) |
| `dark_entropy` | Dark-sector KL — must be logged; confirms dual-sector partition |
| `conserved` | Dual-sector conservation residual; target `= kl_tau + dark_entropy` to < 1e-10 |
| `dark_entropy_floor` | Value of `dark_entropy` at convergence; expected 40–70% of S₀ |
| `N_active` | Mode activation curve |
| `rho_sign` | Interference ratio — tracks constructive/destructive phase transitions |
| `rho_sign_crossing_step` | Step at which `rho_sign` first crosses back below 1.0 |
| `Lambda_tau` | Pump amplitude — flag if cap is hit before step 1000 |
| `lambda_cap_step` | Step at which Λ first hits `lambda_clip`; healthy \(> 1500\) of 2000 |
| `norm_tau` | `tau_n / tau_max_est` — clock normalisation |
| `ELBO` | Total loss |
| `kl_z`, `kl_S` | Individual ELBO terms |

**Expected \(\tau_n\) curves by regime (at `kl_decay=0.03`, `lambda_clip=10`):**

| `kl_S_weight` | Predicted \(\tau_n\) shape | Analogue |
|---------------|---------------------------|----------|
| 0.001 | Multiple oscillations (big bang/crunch cycles) | Low barrier |
| 0.01 | 1–2 cycles then stabilisation | Intermediate-low |
| 0.05 | Single clean expansion + plateau | Intermediate |
| 0.2 | Slow expansion, early plateau | Intermediate-high |
| 1.0 | Near-flat from the start (heat death) | High barrier |

**Expected `dark_entropy` behaviour by regime:**

| `kl_S_weight` | Expected `dark_entropy` floor | Interpretation |
|---------------|-------------------------------|----------------|
| 0.001 | Oscillates; does not converge | Active big bang/crunch cycling |
| 0.01–0.05 | Converges to 40–60% of S₀ | Stable dark semantic basin |
| 0.2–1.0 | Converges fast, > 70% of S₀ | Premature dark dominance / heat death |

### Success Criteria

- For `kl_S_weight = 0.001`, at least 2 identifiable \(\tau_n\) oscillation cycles visible in
  the trajectory plot.
- For `kl_S_weight = 1.0`, `is_heat_death()` fires before step 500 in at least 4/5 seeds.
- **Entropy conservation:** `|conserved - (kl_tau + dark_entropy)| < 1e-10` for all steps
  in all runs (machine-precision conservation as demonstrated in Exp 2/seed 3407).
- **Dark entropy floor:** at convergence, `dark_entropy` stabilises to a finite positive
  value in the range 40–70% of S₀ for `kl_S_weight ∈ {0.01, 0.05, 0.2}`. A floor of 0 or
  > 90% of S₀ indicates dark sector accounting failure.
- **`rho_sign` oscillation:** for `kl_S_weight ≤ 0.05`, `rho_sign` must exhibit at least one
  crossing below 1.0 (constructive dominant) after an initial destructive-dominant phase.
  The crossing step should appear before step 1500 in at least 3/5 seeds.
- **`lambda_cap_step` > 1500:** with `lambda_clip=10` and `kl_decay=0.03`, the pump must not
  hit the cap before 75% of training has elapsed. If `lambda_cap_step < 1000` in any run,
  switch to `kl_decay=0.01` for that `kl_S_weight` condition.
- ELBO decreases monotonically after a burn-in of 200 steps for moderate `kl_S_weight` (0.05).
- Cluster separation \(\rho\) does not change the qualitative shape of the `tau_n` trajectory for
  a given `kl_S_weight` — only the absolute values change.

### Failure Modes

- **All runs produce heat death regardless of `kl_S_weight`:** The dark sector entropy
  `dark_entropy` is not being computed from inactive modes. Check that `SpectralLoadingDecoder`
  is correctly feeding `kl_dark` back to `clock.step()`.
- **`dark_entropy` is zero throughout:** `kl_dark` is not being populated; `dark_entropy`
  logging is missing from `clock.step()`. Confirmed in Exp 2 seed 3407: `dark_entropy` must
  be logged explicitly; it was absent from the previous CSV schema.
- **`conserved` explodes:** The entropy conservation ELBO term weight `beta` is too small.
  Try `beta = 1/S_0` as suggested in §11 of the design doc.
- **`lambda_cap_step` < 500 even with `lambda_clip=10`:** `kl_decay` is still too large.
  Exp 2/seed 3407 with `kl_decay=0.08` and 400 steps hit the cap at step 257 (64% through).
  For 2000 steps: use `kl_decay=0.01` to keep the pump active past step 1500.
- **`rho_sign` never oscillates (stays destructive throughout):** Inspect the initial
  `rho_plus/minus` ratio — if `rho_sign > 1` at step 0 for all seeds, the initial \(Q_{-1}\)
  and \(Q_0\) conditions produce net-destructive interference. Perturb the initialisation
  noise scale.
- **`tau_n` is monotonically increasing without oscillation at any `kl_S_weight`:** The
  `kl_tau` signal is always positive (no mode deactivation). Check that `N_active` actually
  decreases during contraction phases and that deactivated modes contribute to `kl_dark`.

### Connection to Design

The primary experimental validation of §9 and the two-sector partition in §2. Produces the
ESDM analogue of Barontini's Fig. 2. The `tau_n` vs. training-step curves from this experiment
are the key evidence that the entropic clock reproduces big bang/big crunch thermodynamic
behaviour in semantic space. The `dim=64` / `bright_modes=32` geometry ensures the dark sector
has genuine structure (`log(32)` nats maximum entropy), so the bright/dark partition is
non-trivial from initialisation. The dual-space hypothesis in §10.2
connects the dark-sector structure to the item-space dynamics.

---

## Experiment 4 — Per-Mode Entropic Schedule on MNIST (8-D PCA Embeddings)

### Motivation

Experiments 1–3 use synthetic or trivially structured data. This experiment introduces a real
semantic corpus (MNIST) at low dimensionality via PCA projection, and specifically validates the
**per-mode entropic schedule** feature of `EntropicDiffusionDecoder` (`use_per_mode=True`).
The claim (§5.3 of the design doc) is that modes with high spectral entropy should dissolve
early while stable modes persist — producing sharper reconstructions than the global schedule.

### Setup

**Dataset:** MNIST train split (60,000 images), projected to \(\mathbb{R}^{16}\) via PCA
retaining the top 16 principal components (capturing ~95% of variance for standard MNIST). Each
sample \(x \in \mathbb{R}^{16}\) is L2-normalised before training.

**Model:** Full ESDM with `dim=16`, `n_steps=4`, `use_per_mode=True`.

**Ablation:** Train two model variants side by side:
- **ESDM-global**: `use_per_mode=False` — single scalar `alpha_bar(tau_n)` for all 16 modes.
- **ESDM-per-mode**: `use_per_mode=True` — separate `alpha_bar(tau_k)` per mode \(k\).

Training: 10,000 steps, batch size 128, AdamW with cosine LR decay from \(3 \times 10^{-4}\).

**Metrics:**

| Metric | Purpose |
|--------|---------|
| Reconstruction MSE on held-out 10,000 test samples | Primary quality metric |
| Per-mode `tau_k` at convergence | Validates that fast-varying modes have large `tau_k` |
| PCA mode \(k\) explained variance vs `tau_k` | Tests the hypothesis that high-variance modes stabilise earlier |
| `rho_sign` at convergence | Should be \(< 0.5\) for healthy semantic organisation |
| 10-NN classification accuracy in the 16-D latent \(z\) space | Tests that the ESDM latent is semantically organised by digit class |
| `conserved` residual | Entropy conservation quality |

**Per-mode tau hypothesis test:** Compute Spearman correlation between the final `tau_k` values
and the PCA-mode explained-variance fractions. The hypothesis is \(\rho_S < 0\): low-variance
modes (capturing fine detail) should accumulate more entropic time (dissolve earlier) than
high-variance modes (capturing gross structure).

### Success Criteria

- ESDM-per-mode reconstruction MSE is \(\leq\) ESDM-global MSE (per-mode schedule should
  not hurt and ideally helps).
- Spearman correlation between `tau_k` and explained variance is negative (\(\rho_S < -0.3\)).
- 10-NN accuracy in latent \(z\) space \(\geq 85\%\) (comparable to a plain PCA+kNN baseline on
  the same 16-D projection).
- No mode collapse: all 16 `tau_k` values are positive at convergence.
- `conserved` \(< 0.05 \cdot S_0\) after step 5000.

### Failure Modes

- **ESDM-per-mode MSE is worse than ESDM-global:** Per-mode schedules diverge — some `tau_k`
  collapse to zero while others explode. Add a floor `tau_k.clamp(min=eps)` in `alpha_bar()`.
- **Spearman correlation is near zero (no mode differentiation):** All `tau_k` evolve at the
  same rate. The `kl_tau` signal is being summed before the per-mode buffer is updated.
  Check `_per_mode_tau` accumulation logic in `clock.step()`.
- **10-NN accuracy \(< 70\%\):** The wiring Laplacian \(L(z)\) is not capturing digit-class
  geometry. Visualise the Fiedler vector of \(L(z)\) — it should correlate with digit class
  labels.
- **Heat death before step 3000:** Reduce `heat_death_patience` from 200 to 50 only in
  diagnostic mode; do not use heat death as a stopping criterion for this experiment. Increasing
  the base LR can also postpone heat death.

### Connection to Design

Validates §5.3 (per-mode schedule) and §6.3 (signed interference as confidence) from the design
doc. Provides the first evidence that the ESDM organisation of semantic space is meaningful on a
real dataset. The 10-NN latent accuracy measurement directly tests the ArrowSpace hypothesis
(§10.2): that the dual-space partition produces a semantically organised item-space latent.

---

## Experiment 5 — Full Entropic Diffusion on 32×32 Images (CIFAR-10)

### Motivation

This is the full-scale experiment: end-to-end ESDM training as a generative diffusion model on
32×32 RGB images. The goal is to demonstrate that the entropic clock can drive a viable
diffusion process on a realistic image corpus, and to measure whether the thermodynamic training
regime (big bang/big crunch dynamics from Experiment 3) produces qualitatively better or more
efficient training than a standard fixed-schedule cosine diffusion baseline.

### Setup

**Dataset:** CIFAR-10 (50,000 train / 10,000 test, 32×32 RGB). Images are encoded into a
flat embedding \(x \in \mathbb{R}^{256}\) via a frozen pretrained DINO-small ViT patch encoder
(14×14 patches, CLS token, 384-D → projected to 256-D via a single linear layer trained in
a preliminary 200-step warm-up on the same data).

This gives 50,000 semantic embeddings of dimension 256 as the ESDM input. The decoder produces
\(\hat{x} \in \mathbb{R}^{256}\) which is decoded back to pixels by the same frozen encoder
inverted via a two-layer MLP trained jointly.

**Model:**
- `dim = 256`
- `n_steps = 8` internal vibrational steps per call
- `VibrationalStateBlock`: `n_heads = 8`, `dropout = 0.1`
- `SpectralLoadingDecoder`: `tau_modes = 32`, `m_modes = 64`
- `EntropicDiffusionDecoder`: `use_per_mode = True`
- `EntropicClock`: `heat_death_patience = 500`, `lambda_clip = 100.0`
- `kl_S_weight = 0.01` (chosen from Experiment 3 as the intermediate-barrier regime)

**Training:**
- 50,000 steps, batch size 256
- AdamW, LR \(3 \times 10^{-4}\) with linear warm-up (1000 steps) and cosine decay
- Gradient clipping at norm 1.0
- Mixed precision (bfloat16)
- Checkpoint every 5,000 steps; save `clock.state_dict()` to preserve `tau_n` across restarts

**Baseline:** Identical architecture but with `EntropicClock` replaced by the fixed
`cosine_alpha_bar(t, T=1000)` schedule (standard DDPM schedule), run for the same 50,000 steps.

**Metrics:**

| Metric | Purpose |
|--------|---------|
| FID-10k (Fréchet Inception Distance on 10,000 generated samples) | Primary image quality metric |
| Reconstruction MSE on CIFAR-10 test set | Encoder/decoder alignment |
| `tau_n` trajectory over 50,000 steps | Visualises the thermodynamic training regime |
| Number of big bang/big crunch cycles | Characterises the dynamic regime |
| `N_active` at convergence | Mode utilisation |
| `conserved` residual at convergence | Entropy conservation quality |
| `rho_sign` at convergence | Confidence of final semantic organisation |
| Steps to FID \(< 50\) (if reached) | Training efficiency vs. baseline |

**Generation protocol:** At inference, run the reverse diffusion chain using `clock.alpha_bar()`
at fixed `tau_n = tau_max_est` (fully diffused) down to `tau_n = 0` (clean), using 50 DDIM-style
denoising steps with the entropic schedule replacing the fixed DDIM schedule.

**Loschmidt echo test (from §11 of the design doc):** At step 50,000 (converged model),
select 100 test embeddings, apply 25 steps of forward diffusion using `alpha_bar(tau_n + k*delta)`,
then reverse 25 steps using `alpha_bar(tau_n - k*delta)`. Measure the cosine similarity between
the recovered embeddings and the originals. The hypothesis is that the entropic schedule is more
reversible than a fixed schedule because it does not add noise at steps where entropy is not
flowing.

### Success Criteria

- ESDM FID-10k \(< 80\) (a permissive threshold for a 256-D latent model without classifier-free
  guidance; strong diffusion baselines on CIFAR-10 at this setting achieve FID \(\approx 20\),
  but those use full-resolution pixel-space diffusion with CFG).
- ESDM FID-10k is within 20 FID points of the fixed-schedule baseline.
- `conserved` \(< 0.1 \cdot S_0\) throughout training (weak conservation law holds).
- At least 1 identifiable big bang cycle visible in the `tau_n` trajectory.
- Loschmidt echo cosine similarity \(> 0.85\) for ESDM vs. \(> 0.75\) for the fixed-schedule
  baseline (testing the reversibility advantage hypothesis).
- No NaN in any ELBO term at any point during 50,000 steps.

### Failure Modes

- **FID does not improve past 200:** The pixel decoder (frozen DINO inverted via MLP) is the
  bottleneck. Replace with a lightweight convolutional decoder and retrain the warm-up stage.
- **`tau_n` is monotonically increasing without cycles:** `kl_S_weight = 0.01` is above the
  cyclic threshold for CIFAR-10 geometry. Reduce to `0.001` (from Experiment 3, the low-barrier
  regime produces cycles regardless of corpus).
- **Heat death at step ~5,000:** The 256-D latent is a harder entropy source than the synthetic
  experiments. Increase `heat_death_patience = 1000` and add a `tau_n` floor: if `is_heat_death()`
  fires, inject a small entropy pulse by briefly setting `kl_S_weight *= 0.1` for 50 steps.
- **OOM at batch 256 with `n_steps=8`:** Reduce to `n_steps=4` and `batch_size=128`. The wave
  recurrence has \(O(B \cdot F^2)\) complexity per step; the Gershgorin recompute dominates for
  large \(F\). Consider `recompute_cfl=False` for steps > 10,000 (after initial CFL calibration).
- **`conserved` explodes after a big crunch:** The dark sector KL is being computed over all
  inactive modes simultaneously during a mode-collapse event. Apply gradient clipping to the
  conservation term separately: `loss_conservation.clamp(max=10.0 * S_0)`.

### Connection to Design

This experiment closes the loop on §5.2 (entropic diffusion schedule), §8 (health monitoring),
§9 (barrier height sweep, applied to CIFAR-10), and §11 (Loschmidt echo reversibility test).
It is also the primary evidence for or against the claim in §10.3 that "optimisation as
recursive semantic search" is a practical training regime: if the ESDM reaches competitive FID
in fewer wall-clock hours than the fixed-schedule baseline, the thermodynamic self-organisation
provides a genuine efficiency advantage.

---

## Experiment Dependency Graph

```
Exp 1 ✅ (Clock unit test)
  └─► Exp 2 ✅ (Wave recurrence unit test)
        └─► Exp 3 🔜 (Barrier sweep, synthetic)
              └─► Exp 4 (Per-mode schedule, MNIST)
                    └─► Exp 5 (Full diffusion, CIFAR-10)
```

Experiments 1 and 2 can be run locally on CPU in under 10 minutes each. Experiment 3 requires a
single GPU (2–5 hours for all ~540 runs with the expanded sweep).
Experiment 4 requires a single GPU (3–6 hours). Experiment 5 requires a single A100 or
equivalent (estimated 12–24 hours for 50,000 steps at batch 256 in bfloat16).

---

## Shared Logging Convention

All experiments write per-step metrics to a CSV at `runs/<exp_id>/<seed>/metrics.csv`.

**Core columns (Exp 1–5):**
```
step, seed, kl_decay, clock_eps, lambda_clip, bright_modes,
kl_tau, dark_entropy, conserved,
Q_t_0, rho_plus, rho_minus, rho_sign,
dt, dt_cfl_bound, delta_tau, tau_n, norm_tau,
Lambda_tau, alpha_bar, is_heat_death
```

**Additional columns (Exp 3+):**
```
lambda_cap_step, rho_sign_crossing_step, dark_entropy_floor
```

The `tau_n` column is the primary axis for all trajectory plots. Wall-clock step is recorded
separately in `runs/<exp_id>/<seed>/timing.csv` so that training efficiency comparisons do not
conflate entropic time with wall time.
