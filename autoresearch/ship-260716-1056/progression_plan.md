# Issue #29 Progression Plan (v0.3.0)

## What's done (v0.3.0)

| Step | Status | Key result |
|---|---|---|
| lambda_clip derivation (0.1*F) | DONE | 100.0 -> 25.6, nef improved 64->34 |
| U_q-constrained init | DONE | Noise stays in spectral subspace |
| Q3 invalidation | DONE | DDPM bypass is a category error |
| Q-space score-approximation sampler | DONE | Non-zero embeddings, correct marginal stats |
| State leak fix (Q_tm1, rho) | DONE | Sampler now fully reproducible |

## Where we are

The Q-space sampler produces embeddings with correct scale (norm=1.0) and
marginal statistics (std=0.0625) but they're off-manifold (NN ratio=75 vs
ideal 1.0). The bottleneck has shifted from **sampler architecture** to
**model quality** — the Lambda/H imbalance means the model can't produce
good Q predictions from noised states.

## Progression (ordered by dependency)

### Phase 2 — Fix Lambda/H imbalance (THE BLOCKER)

The pump still dominates H throughout training. `pi_contraction_ratio`
decays from 1.54 -> 0.089, meaning H becomes inert by mid-training. No
sampler will produce on-manifold embeddings until H and Lambda are the
same order of magnitude throughout.

**Step 2a — Ramp Lambda on alpha_bar (not just clip)**
- Gate Lambda by alpha_bar so the pump ramps up gradually instead of
  starting at full clip. Formula: `Lambda_eff = Lambda * (1 - alpha_bar)`
  or similar schedule that makes Lambda small at high alpha_bar (start of
  training) and larger as modes stabilize.
- This is a clock.py change, not a model architecture change.

**Step 2b — Initialize eta larger or make eta adaptive**
- The derived `eta = 0.1*F / (2*lambda_max)` gives H a fighting chance at
  init but it decays as Q shrinks. Consider making eta proportional to
  `1/||Q||` so it scales with the state magnitude.
- Alternatively: make sigma_sq (the diffusion coefficient in H) a learned
  parameter that the optimizer can increase to strengthen H.

**Step 2c — Verify nef oscillates around 0**
- Target: `net_energy_flow` should oscillate +/- around 0, not stay
  persistently positive (which means pump dominates).
- Success: `pi_contraction_ratio` stays in [0.01, 0.3] throughout training,
  not just briefly.

### Phase 3 — Metric redefinition

**Step 3a — Split conservation tautology**
- Replace the tautological `conservation` (15pts, always passes) with:
  - `energy_balance` (10pts): `|net_energy_flow| / ||Q||^2` should be small
  - `diversity` (5pts): variance of generated embeddings should match real

**Step 3b — Replace recon_decrease with recon_absolute**
- Current metric rewards tiny decreases in an already-small quantity.
- Replace with: `recon` must be below a threshold at alpha_bar > 0.5.

### Phase 1 (revisit) — Validate generation with fixed model

Once Phase 2 is done, re-run `validate_generation.py --sampler qspace`:
- Target: NN ratio < 5.0 (currently 75)
- If the model can produce good Q predictions, the DDIM-style update
  should converge toward the real data manifold.

### Phase 4 — Ablations (#27)

Once generation works:
- Step-count sweep (does more steps improve manifold coverage?)
- Non-spectral baseline (replace L_F with identity — does H contribute?)
- Diversity/quality decomposition

### Phase 5 — New tracks

- #24: 2.5-D dual-space track (unblocked, low priority)
- #25: Audio generation (blocked on working generation + audio loader)
- Experiment 4 Condition C: `rectify_lifting` toggle (blocked on #29)

## Quick wins (can start now, no dependency)

1. Fix k-NN index bug in validate_generation.py (train_labels shorter
   than train_data)
2. Add FID computation to validate_generation.py (needs pytorch-fid or
   torchmetrics)
3. Train a PixelDecoder on (embedding, image) pairs from CIFAR10DINOLoader
   (requires retaining raw images in the loader)
