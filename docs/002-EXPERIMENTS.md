# 001 — Experiments: 2.5-D vs 3D Diffusion on Minimal Surfaces

> Replaces previous experiment plans. All spectral/graph computation uses
> `arrowspace` (`pip install arrowspace`); test conventions follow
> `pyarrowspace/tests/`.

## Goal

Determine whether a 2.5-D dual-space diffusion model (2D item coordinate +
graph-Laplacian feature signal, wired via `arrowspace`) can match the
generative quality of a full 3D volumetric diffusion model on physical
surfaces, without computing Jacobian determinants.

## Experiment 1 — Baseline replication (3D)

**Purpose**: establish a correct, well-behaved 3D score-based diffusion
model as the control condition.

- **Data**: synthetic point clouds sampled from analytic minimal surfaces
  (Enneper patch, helicoid, catenoid), 3D coordinates, known zero mean
  curvature ground truth.
- **Model**: `models/score3d.py`, MLP score network, denoising score
  matching (DSM) loss.
- **Sampler**: `sampling/euler_maruyama.py` in 3D mode.
- **Success criterion**: generated point clouds have discrete mean curvature
  within a defined tolerance of zero, and Chamfer distance to held-out real
  samples below a baseline threshold.

## Experiment 2 — 2.5-D dual-space model

**Purpose**: implement and sanity-check the treatment condition.

- **Graph construction**: `arrowspace.build_graph` on the same point sets
  used in Experiment 1, following `pyarrowspace/tests` fixture patterns
  (fixed seed, fixed `k`).
- **Item coordinate \(z\)**: 2D spectral embedding (leading non-trivial
  Laplacian eigenvectors) obtained via `arrowspace`.
- **Feature signal \(f\)**: taumode / Rayleigh-quotient score per node,
  from `graph.taumode_score()`.
- **Model**: `models/score25d.py`, using the update equations in
  `001-IMPLEMENTATION-PLAN.md` §4.2.
- **Success criterion**: the graph-diffusion step on \(f\) reproduces the
  smoothing behavior of `arrowspace`'s own taumode computation on held-out
  synthetic signals (regression test against `arrowspace` output directly).

## Experiment 3 — Head-to-head comparison

**Purpose**: the central test of the ESDM hypothesis.

| Metric | Definition | Pass condition |
|---|---|---|
| Chamfer distance | Nearest-neighbor distance, generated vs. reference point sets | 2.5-D within X% of 3D baseline (X to be fixed after pilot run) |
| Rayleigh/λτ energy gap | \|R(f_gen) − R(f_real)\|, via `arrowspace.taumode_score` | 2.5-D energy gap ≤ 3D baseline's implicit equivalent |
| Mean curvature error | Discrete mean curvature vs. analytic zero | 2.5-D within same order of magnitude as 3D baseline |
| Wall-clock / peak memory | Training + sampling time, per condition | 2.5-D shows measurable reduction (expected win, since no determinant/volume term is computed) |

**Procedure**:
1. Train both models on identical train/test splits of the same synthetic
   surfaces (same seed).
2. Sample N surfaces from each model.
3. Compute all four metrics per model.
4. Report mean ± std over multiple training seeds (minimum 3 seeds).

## Experiment 4 — Ablation: item coordinate alone vs. dual space

**Purpose**: isolate whether the feature-space Laplacian signal is doing
real work, or whether the 2D item coordinate alone is sufficient.

- **Condition A**: full dual-space model (\(z\), \(f\)) as in Experiment 2.
- **Condition B**: 2D item coordinate \(z\) only, no feature-signal
  conditioning, no Laplacian diffusion term.
- **Metric**: Chamfer distance and mean curvature error, Condition A vs. B,
  both vs. the 3D baseline from Experiment 1.
- **Interpretation**: if A closes most of the gap to 3D while B does not,
  this is direct evidence that graph-wired feature information — not merely
  a lower-dimensional coordinate — substitutes for the missing third
  dimension, supporting the ESDM dual-space reunification claim.

## Experiment 5 — Spectral stability under subsampling

**Purpose**: validate that the graph-wiring representation used by the
2.5-D model is robust, per the Graph Wiring paper's validation protocol.

- **Procedure**: re-wire the graph on random subsamples of the point cloud
  (e.g. 90%, 75%, 50% of points) using `arrowspace.build_graph`.
- **Metric**: λτ tail stability (rank correlation of taumode scores across
  subsample levels), using the same Spearman/Kendall/NDCG@10 style metrics
  used in prior `arrowspace` evaluations.
- **Pass condition**: λτ tail rank correlation remains high (comparable to
  published `arrowspace` benchmarks) across subsampling levels, indicating
  the 2.5-D model's feature signal does not depend on a fragile graph.

## Experiment 6 — Compute cost comparison

**Purpose**: quantify the practical benefit of avoiding volume computation.

- **Metrics**: training time per epoch, sampling time per batch, peak GPU/CPU
  memory, for 3D baseline vs. 2.5-D model, at matched model capacity
  (parameter count as close as feasible).
- **Reporting**: table plus a short note on how much of any speedup is
  attributable to lower dimensionality vs. to reuse of `arrowspace`'s sparse
  Laplacian operations instead of dense volumetric computation.

## Reporting format

Each experiment produces:
- A CSV of raw per-seed metric values.
- A summary table (mean ± std) matching the structure above.
- Where applicable, a comparison plot (Chamfer distance and curvature error
  vs. training step, 3D vs. 2.5-D overlaid).

## Out of scope for this experiment set

- Real-world scanned mesh data (deferred to a follow-up track).
- Any change to `arrowspace` internals — all issues found during
  implementation are filed upstream against `pyarrowspace`, not patched
  locally.
