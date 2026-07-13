# 002 — Experiments: 2.5-D vs 3D Diffusion on Minimal Surfaces

> Replaces previous experiment plans. All spectral/graph computation uses
> `arrowspace` (`pip install arrowspace`); test conventions follow
> `pyarrowspace/tests/`.
>
> **API correction (2026-07-07):** the `arrowspace==0.26.2` Rust bindings do
> not expose `build_graph`, `graph.laplacian()`, or `graph.taumode_score()`.
> All such calls below are routed through the adapter in
> `esdm/graphs/wiring.py` (see `001-IMPLEMENTATION-PLAN.md` §3.1–3.2). The
> Rayleigh/λτ score is computed locally in the adapter until a native
> `taumode_score` lands upstream.
>
> **Update (2026-07-13, issue #27):** added Experiments 7–9 to close
> validation gaps identified by comparison with Un-0 (Unconventional AI).
> Un-0 uses Kuramoto coupled oscillators as its generative backbone; ESDM
> uses a string/spring system constructed on the F×F feature-space Laplacian
> $L_F$ (ArrowSpace graph wiring), driven by the entropic clock $\tau$.
> The two architectures share the same class of resonant physical dynamics
> but differ in substrate: Un-0 learns a dense pairwise phase-coupling matrix
> over oscillators; ESDM learns a WHT spectral gate $g \in \mathbb{R}^F$ that
> selects which sequency bands of $L_F$ propagate into the wave recurrence.
> The experiments below are adapted to ESDM's design — no Kuramoto equations
> are introduced.

## Goal

Determine whether a 2.5-D dual-space diffusion model (2D item coordinate +
graph-Laplacian feature signal, wired via `arrowspace`) can match the
generative quality of a full 3D volumetric diffusion model on physical
surfaces, without computing Jacobian determinants; and validate that the
string/spring spectral dynamics of ESDM are doing real computational work,
not merely providing a richer initialisation.

---

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

---

## Experiment 2 — 2.5-D dual-space model

**Purpose**: implement and sanity-check the treatment condition.

- **Graph construction**: `arrowspace.build_graph` on the same point sets
  used in Experiment 1, following `pyarrowspace/tests` fixture patterns
  (fixed seed, fixed `k`).
- **Item coordinate $z$**: 2D spectral embedding (leading non-trivial
  Laplacian eigenvectors) obtained via `arrowspace`.
- **Feature signal $f$**: taumode / Rayleigh-quotient score per node,
  from `graph.taumode_score()`.
- **Model**: `models/score25d.py`, using the update equations in
  `001-IMPLEMENTATION-PLAN.md` §4.2.
- **Success criterion**: the graph-diffusion step on $f$ reproduces the
  smoothing behavior of `arrowspace`'s own taumode computation on held-out
  synthetic signals (regression test against `arrowspace` output directly).

---

## Experiment 3 — Head-to-head comparison

**Purpose**: the central test of the ESDM hypothesis.

| Metric | Definition | Pass condition |
|---|---|---|
| Chamfer distance | Nearest-neighbor distance, generated vs. reference point sets | 2.5-D within X% of 3D baseline (X to be fixed after pilot run) |
| Rayleigh/λτ energy gap | $\|R(f_\text{gen}) - R(f_\text{real})\|$, via `arrowspace.taumode_score` | 2.5-D energy gap ≤ 3D baseline's implicit equivalent |
| Mean curvature error | Discrete mean curvature vs. analytic zero | 2.5-D within same order of magnitude as 3D baseline |
| Wall-clock / peak memory | Training + sampling time, per condition | 2.5-D shows measurable reduction (expected win, since no determinant/volume term is computed) |

**Procedure**:
1. Train both models on identical train/test splits of the same synthetic
   surfaces (same seed).
2. Sample N surfaces from each model.
3. Compute all four metrics per model.
4. Report mean ± std over multiple training seeds (minimum 3 seeds).

---

## Experiment 4 — Ablation: item coordinate alone vs. dual space

**Purpose**: isolate whether the feature-space Laplacian signal is doing
real work, or whether the 2D item coordinate alone is sufficient.

- **Condition A**: full dual-space model ($z$, $f$) as in Experiment 2.
- **Condition B**: 2D item coordinate $z$ only, no feature-signal
  conditioning, no Laplacian diffusion term.
- **Condition C** _(new, issue #27)_: dual-space with **unsigned (rectified)**
  feature signal $|f|$ — Laplacian action is preserved but all destructive
  interference (negative amplitudes in the spring/string resonance) is
  suppressed. This isolates the contribution of signed spectral coefficients
  in $L_F$ without removing the dual-space structure entirely.
  - In ESDM's string/spring framing: Condition C corresponds to a system
    where all springs are purely repulsive (positive restoring force only),
    removing the possibility of anti-phase mode cancellation that the design
    doc identifies as a prediction-confidence signal ($\rho_- > \rho_+$ in
    the `SignedDensityMatrix`).
- **Metric**: Chamfer distance, mean curvature error, and mode-separation
  metric (std of per-sample Rayleigh quotients) for Conditions A, B, C vs.
  the 3D baseline.
- **Interpretation**: if A outperforms C on mode separation, signed
  interference in the feature signal is contributing real discriminative
  capacity, not just acting as a sign convention. If A ≈ C, the unsigned
  ablation is sufficient and the quantum-interference framing should be
  revisited.

---

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

---

## Experiment 6 — Compute cost comparison

**Purpose**: quantify the practical benefit of avoiding volume computation.

- **Metrics**: training time per epoch, sampling time per batch, peak GPU/CPU
  memory, for 3D baseline vs. 2.5-D model, at matched model capacity
  (parameter count as close as feasible).
- **Reporting**: table plus a short note on how much of any speedup is
  attributable to lower dimensionality vs. to reuse of `arrowspace`'s sparse
  Laplacian operations instead of dense volumetric computation.

---

## Experiment 7 — Frozen WHT gate (reservoir) vs. learned gate _(new, issue #27)_

**Purpose**: prove that the *learned* spectral gate $g$ is doing real
computational work, and that the ESDM string/spring dynamics are not merely
a rich random-feature expander.

**Design rationale**: Un-0 validates its Kuramoto backbone by freezing the
coupling matrix at random initialisation and comparing FID against the
trained model. ESDM's analogous control is to freeze the WHT gate $g$ at
its initialisation value ($g = \mathbf{1}$, all sequency bands equally
weighted) and compare against the co-trained gate.

- **Condition A** (treatment): full ESDM with learned $g$, co-trained
  end-to-end.
- **Condition B** (reservoir / frozen): $g$ fixed at $\mathbf{1}$ throughout
  training; all other parameters trained normally. The Laplacian action is
  still applied — all string/spring modes propagate equally — but no
  frequency selection is learned.
- **What the comparison shows**:
  - If A significantly outperforms B: the model is learning a meaningful
    frequency selection over the feature-space Laplacian — the pass-band
    behaviour described in `00-DESIGN_PREAMBLE.md` §WHT Spectral Filter is
    real, not incidental.
  - If A ≈ B: the gate is not contributing beyond its architectural role as
    a scaling layer; the Laplacian diffusion itself (regardless of gate) is
    doing the work. This is still a positive result for ESDM but changes
    the claim about what the WHT filter adds.
- **Metrics**: Chamfer distance, Rayleigh/λτ energy gap, fitness score
  (active phase only), `g_entropy` trajectory.
- **Procedure**: train both conditions on identical splits, same seed, same
  number of steps. Report mean ± std over 3 seeds.

---

## Experiment 8 — Unroll-depth sweep for the reverse diffusion _(new, issue #27)_

**Purpose**: validate that the entropic clock and spectral diffusion are
adding real capacity across reverse steps, not just at the terminal state.

**Design rationale**: Un-0 shows that generation quality (FID) improves
monotonically with the number of oscillator integration steps. In ESDM, the
analogous parameter is the number of reverse diffusion steps in
`EntropicDiffusionDecoder` (controlled by the discretisation of
$\tau_\text{max}$). If quality does not improve with more steps, the diffusion
trajectory is not contributing beyond a single-shot prediction.

- **Sweep**: run the reverse-SDE at $N \in \{1, 2, 4, 8, 16, 32\}$ steps,
  holding all other hyperparameters fixed.
- **Metrics at each step count**:
  - Chamfer distance (primary quality metric)
  - Rayleigh/λτ energy gap (spectral fidelity)
  - Wall-clock time per sample (cost curve)
- **Termination check**: verify that the reverse process reaches $\tau = 0$
  cleanly at every step count (no NaN, no energy explosion in $Q_t$).
- **Expected outcome**: quality improves with step count up to a saturation
  point; the location of saturation informs the default $N$ for the
  generation pipeline.
- **Additional diagnostic**: at each step count, measure the `conserved`
  residual $|S_\text{bright} + S_\text{dark} - S_0|$ at the terminal step.
  A well-behaved run should converge conservation regardless of step count.

---

## Experiment 9 — Diversity / quality decomposition _(new, issue #27)_

**Purpose**: characterise *what* the spectral string/spring dynamics are
contributing — basin coverage (recall / diversity) vs. sample sharpness
(precision / quality) — in line with the dual-space hypothesis that
Laplacian-guided diffusion should improve semantic basin coverage.

**Design rationale**: aggregate metrics (Chamfer distance, FID) cannot
distinguish a model that generates diverse but coarse samples from one that
generates sharp but mode-collapsed samples. ESDM's design motivation —
that $L_F$ encodes semantic basin structure and WHT-filtered diffusion
follows basin geometry — predicts improved diversity relative to a
non-spectral baseline, not merely improved per-sample fidelity.

- **Metrics**:
  - **Coverage** (recall analogue): for each reference sample, measure
    whether at least one generated sample is within Chamfer threshold $\delta$.
    Report fraction of reference set covered.
  - **Precision** (quality analogue): for each generated sample, measure
    whether it is within $\delta$ of at least one reference sample. Report
    fraction of generated set within manifold.
  - **Mode-separation metric**: std of per-sample Rayleigh quotients
    $R(f_i) = f_i^\top L_F f_i / f_i^\top f_i$ across the generated set.
    Higher std indicates the generator is visiting multiple spectral regimes
    (multiple semantic basins), not collapsing to one mode.
- **Conditions compared**:
  - Full ESDM (learned gate, trained dynamics)
  - Frozen-gate ESDM (Condition B from Experiment 7)
  - Non-spectral MLP baseline (no $L_F$ term, matched parameter count)
- **Interpretation**: if full ESDM has higher coverage and mode-separation
  than frozen-gate and non-spectral baseline, this is direct evidence that
  the string/spring spectral dynamics are improving semantic basin coverage
  as the dual-space hypothesis predicts.

---

## Reporting format

Each experiment produces:
- A CSV of raw per-seed metric values.
- A summary table (mean ± std) matching the structure above.
- Where applicable, a comparison plot (Chamfer distance and curvature error
  vs. training step, 3D vs. 2.5-D overlaid).
- For Experiments 7–9: an additional column in the summary table recording
  the `conserved` residual at the terminal step, so conservation law
  compliance is reported alongside quality metrics.

## Out of scope for this experiment set

- Real-world scanned mesh data (deferred to a follow-up track).
- Any change to `arrowspace` internals — all issues found during
  implementation are filed upstream against `pyarrowspace`, not patched
  locally.
- Kuramoto-style dense pairwise coupling matrices — ESDM's spring/string
  substrate operates on $L_F$ (sparse, graph-wired) not on a learned N×N
  oscillator coupling. Comparisons to Un-0 are analytical, not
  implementational.
