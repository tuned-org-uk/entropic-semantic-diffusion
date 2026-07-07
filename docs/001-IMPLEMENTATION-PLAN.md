# 001 — ESDM: PyTorch Implementation Plan

> **Status:** Implementation-ready design  
> **Depends on:** `docs/00-DESIGN.md`, `000-DIFFUSION-ON-PHYSICAL-SURFACES.md`  
> **Integrates:** ArrowSpace dual-space fusion analysis, Barontini 2026, SPIN/GW spectral indexing, 2.5-D hypothesis  
> **Target:** PyTorch ≥ 2.3, Python ≥ 3.12

# 001 — Implementation Plan: 2.5-D Dual-Space Diffusion Track

> Status: DRAFT — new track added alongside existing ESDM design (see `00-DESIGN.md`).
> Dependency: all Laplacian / graph-wiring computation MUST reuse `arrowspace`
> (`pip install arrowspace`, https://github.com/tuned-org-uk/pyarrowspace).
> No custom Laplacian, Rayleigh quotient, or spectral graph code should be
> reimplemented in this repo — call into `arrowspace` and, where API gaps exist,
> contribute upstream to `pyarrowspace` rather than forking logic locally.
> Usage patterns should mirror `tests/` in the `pyarrowspace` repository.

## 1. Motivation

ESDM's core theoretical claim is that a continuous 3D generative diffusion
process over a physical-surface item space can be replaced by a dual-space
representation:

- **Item space**: a low-dimensional (≤ 2D) coordinate per data point,
  typically a spectral embedding derived from the wired graph.
- **Feature space**: a signal defined on the same graph, whose smoothness is
  captured by the Rayleigh quotient / λτ score computed by `arrowspace`.

The working hypothesis is that this **dual (item, feature) representation**
carries enough information to approximate a 3D volumetric diffusion process
without ever computing a Jacobian determinant, i.e. without reasoning about
volume at all. This document defines the concrete implementation track for
building, training, and comparing this "2.5-D" diffusion model against a
full 3D baseline.

## 2. Scope of this track

In scope:
- A PyTorch reference implementation of a 3D score-based diffusion baseline.
- A PyTorch reference implementation of the 2.5-D dual-space diffusion model,
  using `arrowspace` for all graph/Laplacian/Rayleigh-quotient operations.
- A shared synthetic minimal-surface data generator (ground-truth surfaces
  with known analytic curvature).
- A comparison harness (metrics, ablations) — detailed in `001-EXPERIMENTS.md`.

Out of scope (future tracks):
- Real-world 3D scan / mesh datasets.
- Production-grade training infrastructure (multi-GPU, distributed).
- Integration with the CVE-search or LLM-embedding use cases of `arrowspace`.

## 3. Dependency: `arrowspace` usage contract

All graph construction and spectral computation in this repo goes through
`arrowspace`. Concretely:

```python
import arrowspace as asp

# Graph construction from a point set / feature matrix (mirrors pyarrowspace/tests)
graph = asp.build_graph(features, k=k_neighbors, sampler="simple_random")

# Graph Laplacian (sparse) — never hand-rolled
L = graph.laplacian(normalized=True)

# Rayleigh quotient / taumode (λτ) score per item or signal
score = graph.taumode_score(signal, tau=tau)
```

Rules for this track:

1. Any place the implementation plan below says "graph Laplacian" or
"Rayleigh quotient," it means the corresponding `arrowspace` call, not a
local NumPy/PyTorch reimplementation.
2. Sparse matrix outputs from `arrowspace` are converted to
`torch.sparse_coo_tensor` (or CSR) once per graph build, then reused across
training steps — no per-step re-wiring.
3. If `arrowspace` lacks a needed primitive (e.g. differentiable
Laplacian-vector product for backprop through wiring), this is logged as
an open issue against `pyarrowspace` before any local workaround is
written, and the workaround is clearly marked `# TEMPORARY: upstream issue #<n>`.
4. Test patterns for exercising `arrowspace` in this repo should follow the
structure used in `pyarrowspace/tests/` (fixture graphs, deterministic
seeds, taumode score assertions).

## 4. Update equations

### 4.1 Baseline: 3D volumetric diffusion (reverse SDE)

$$
x_{t-\Delta t} = x_t + \frac{\sigma^2}{2}\, s_\theta(x_t, t)\,\Delta t
  + \sigma\sqrt{\Delta t}\,\epsilon, \qquad \epsilon \sim \mathcal N(0, I_3)
$$

$s_\theta: \mathbb R^3 \times [0, T] \to \mathbb R^3$ is a standard MLP/
PointNet-style score network trained with denoising score matching directly
on 3D coordinates. This is the control condition: it implicitly encodes
volume information through the dimensionality of the score field itself.

### 4.2 2.5-D dual-space diffusion

State is split into a 2D item coordinate $z_t$ and a feature signal $f_t$
defined on the wired graph (computed via `arrowspace`):

$$
z_{t-\Delta t} = z_t + \eta\, \nabla_z R(f_t)\, \Delta t
  + \sigma\sqrt{\Delta t}\, \epsilon_z, \qquad \epsilon_z \sim \mathcal N(0, I_2)
$$

$$
f_{t-\Delta t} = f_t - \frac{\sigma^2 \Delta t}{2}\, L_F f_t
$$

where:

- $L_F$ is the graph Laplacian returned by `graph.laplacian()` (arrowspace).
- $R(f) = f^\top L_F f / f^\top f$ is the Rayleigh quotient, computed via
`graph.taumode_score()` (arrowspace), which converges to continuum
Dirichlet energy under the graph-wiring convergence result.
- $\nabla_z R$ is the learned score analogue: a 2D gradient produced by a
small score network conditioned on $f_t$, never requiring a Jacobian
determinant.

This is the treatment condition under test.

## 5. Component breakdown

| Component | Description | Owner module |
| :-- | :-- | :-- |
| `data/minimal_surfaces.py` | Synthetic minimal-surface generators (Enneper, helicoid, catenoid patches) with analytic curvature ground truth | new |
| `graphs/wiring.py` | Thin wrapper around `arrowspace.build_graph`, caching sparse Laplacian as `torch.sparse` tensor | new, depends on `arrowspace` |
| `models/score3d.py` | Baseline 3D score network + DSM loss | new |
| `models/score25d.py` | 2.5-D dual-space score network + graph-diffusion step (uses `graphs/wiring.py`) | new |
| `sampling/euler_maruyama.py` | Shared reverse-SDE sampler, parameterized by dimensionality | new |
| `eval/metrics.py` | Chamfer distance, Rayleigh/λτ energy gap, discrete mean curvature error | new |
| `eval/harness.py` | Orchestrates training + sampling + metrics for both conditions | new |

## 6. Milestones

1. **M1 — Data + graph plumbing**: synthetic minimal-surface generator;
`arrowspace`-backed graph wiring wrapper with unit tests mirroring
`pyarrowspace/tests` patterns.
2. **M2 — 3D baseline**: train and sample the volumetric score model; sanity
check against analytic curvature (should approach a minimal surface as
training converges).
3. **M3 — 2.5-D model**: implement dual-space score network and graph
diffusion step; verify the Laplacian-smoothing step matches `arrowspace`
taumode score behavior on held-out synthetic signals.
4. **M4 — Comparison harness**: run both models on identical splits, produce
the metrics table and ablations defined in `001-EXPERIMENTS.md`.
5. **M5 — Spectral stability check**: re-wire graphs under subsampling and
confirm λτ tail stability, per the validation protocol in the Graph
Wiring paper.

## 7. Open risks / assumptions

- The conformal-gauge assumption linking Dirichlet minimization to minimal
surfaces is imposed, not derived, in the current graph-wiring theory; this
track treats it as a hypothesis to validate empirically, not a given.
- `arrowspace`'s current API may not expose a differentiable Laplacian path
suitable for backprop through the wiring step itself; if so, the wiring
graph is treated as fixed per training run rather than re-optimized
end-to-end (documented as a limitation, not silently patched).
