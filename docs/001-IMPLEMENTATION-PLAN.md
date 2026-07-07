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

### 3.1 Real `arrowspace` API (verified against `arrowspace==0.26.2`)

The installed Rust bindings expose **three** top-level types — none of which
match the convenience surface assumed in earlier drafts of this plan
(`build_graph`, `graph.laplacian()`, `graph.taumode_score()` do **not**
exist). The migration must therefore go through a thin adapter in
`esdm/graphs/wiring.py` that maps the intended interface below onto the real
API:

```python
import arrowspace as asp

# --- Graph construction (real API) ---
builder = (asp.ArrowSpaceBuilder()
           .with_seed(seed)
           .with_sampling("simple_random")        # TEMPORARY: verify sampler name upstream
           .with_cluster_max_clusters(k))         # k-NN analogue; verify mapping upstream
space: asp.ArrowSpace = builder.build_full(features)   # or .build / .build_and_store

# --- Sparse graph Laplacian (real API) ---
gl: asp.GraphLaplacian = space._graph_laplacian()       # TEMPORARY: access path to confirm
L_csr = gl.to_csr()                                     # scipy-style CSR; no `normalized=` flag
L_dense = gl.to_dense()
nnodes, shape = gl.nnodes, gl.shape

# --- Spectral / retrieval primitives that DO exist ---
lambdas_sorted = space.lambdas_sorted()                 # sorted eigenvalue-like spectrum
items = space.search(query, k=k)                        # k-NN retrieval (item-space)
motives = space.spot_motives_eigen(...)                 # eigen-mode motif spotting
```

**Primitives the plan needs but `arrowspace` does not expose** (each filed
upstream against `pyarrowspace` and locally implemented in `graphs/wiring.py`
with a `# TEMPORARY: upstream issue #<n>` marker, per rule 3 below):

| Intended primitive | Real-API status | Adapter action |
| :-- | :-- | :-- |
| `build_graph(features, k, sampler)` | missing | wrap `ArrowSpaceBuilder().with_*(...).build_full(...)` |
| `graph.laplacian(normalized=True)` | partial — `GraphLaplacian.to_csr()` exists, no normalised form | normalise in the adapter: $L_{sym} = D^{-1/2} L D^{-1/2}$ from the CSR |
| `graph.taumode_score(signal, tau)` | **missing** — no Rayleigh/λτ score primitive | compute $R(f) = f^\top L f / f^\top f$ locally in the adapter (this is the core 2.5-D score); file upstream issue for a native, differentiable implementation |
| leading Laplacian eigenvectors (for $z$ embedding) | `lambdas_sorted()` exists; eigenvector accessor not confirmed | eigendecompose `to_dense()` with `torch.linalg.eigh` in the adapter until an upstream accessor lands |

### 3.2 Intended interface (adapter contract)

The rest of this plan codes against the following surface, which
`esdm/graphs/wiring.py` is responsible for providing on top of §3.1:

```python
from esdm.graphs.wiring import WireGraph   # the adapter

g = WireGraph.build(features, k=k_neighbors, seed=3407)   # wraps ArrowSpaceBuilder
L_torch = g.laplacian(normalized=True)                    # -> torch.sparse_coo_tensor, cached
score = g.rayleigh_quotient(f)                            # -> Tensor; the λτ / Dirichlet energy proxy
z = g.spectral_embedding(n_components=2)                  # -> leading non-trivial eigenvectors
```

Rules for this track:

1. Any place the implementation plan below says "graph Laplacian" or
"Rayleigh quotient," it means the corresponding **adapter** call in
`esdm/graphs/wiring.py` (§3.2), which itself wraps the real `arrowspace`
API (§3.1). Only primitives that genuinely do not exist upstream
(currently: the Rayleigh/λτ `taumode_score` and a normalised-Laplacian
flag) are implemented in the adapter, and each carries a
`# TEMPORARY: upstream issue #<n>` marker.
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
