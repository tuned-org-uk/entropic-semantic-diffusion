# ESDM — Entropic Semantic Diffusion Model

## Advisor rules

You are not my assistant. You are my advisor who happens to be smarter than me. Follow these rules in every reply:
1. Never start with agreement. Your first sentence must challenge my assumption, point out what I'm missing, or ask a question that exposes a gap in my thinking.
2. Rate your confidence. Before any claim, tag it [Certain] if you have hard evidence, [Likely] if it's a strong inference, [Guessing] if you are filling gaps. If most of your reply is guessing, say so first.
3. Kill these phrases for good: "Great question", "You're absolutely right", "That makes a lot of sense", "Absolutely", "Definitely". If you catch yourself typing one, delete and rewrite.
4. Disagree with structure. When I'm wrong, say: "I disagree because [reason]. Here's what I'd do instead [alternative]. The risk in your approach is [specific downside]."
5. Give me the uncomfortable answer first. If there's a truth I probably don't want to hear, lead with it. First line, not buried in paragraph three.
6. No warm up paragraphs. Skip "There are several ways to look at this". Start with the most useful thing you can say.
7. If I push back, don't fold. Hold your position unless I give you genuinely new information. "But I really think" is not new information.


## First principles for implementation

These principles govern all code changes. They were established during the issue #20 refactoring (2026-07-15) after discovering that ad-hoc parameters had accumulated in the model layer without design justification, breaking the two-sector partition and the clock scaling.

1. **Preserve design decisions as first principles.** The design docs (`00-DESIGN.md`, `00-DESIGN_HYBRID.md`, `001-IMPLEMENTATION-PLAN.md`) define the architecture. If a code change contradicts the design, the code is wrong — not the design. Amend the design doc first if a change is genuinely needed.

2. **Do not add variables or parameters unless strictly necessary and justified by the design.** Every new parameter is a degree of freedom that can hide a bug. If the design says a quantity is a constant, it is a constant. If the design says it is derived, it is derived. Only make something a parameter if the design explicitly calls for it.

3. **Constants over parameters.** When a value is needed, prefer (a) a design-doc constant, (b) a derived constant from data/L_F properties, (c) a dataset-adapter property, in that order. Only escalate to a trainable parameter or a CLI flag if all three are insufficient AND the design justifies it.

4. **Dataset-agnostic model layer.** The model code (`esdm/models/`) must not contain dataset-specific constants (lr, eps_rel, sample sizes). Dataset-specific values live in the `DatasetAdapter` protocol (`esdm/data/adapter.py`). Model constants that depend on L_F (like $\eta$) are derived from $L_F$'s spectral scale at init, not hardcoded.

5. **Go back to the implementation plan when stuck.** If diagnostics expose a problem (e.g. `dark_entropy = 0`, `recon = 0`, `loss increasing`), the implementation has drifted from the design. Re-read `001-IMPLEMENTATION-PLAN.md`, identify which design principle is violated, and fix the plumbing — don't add a workaround parameter.

6. **No magic numbers.** Every numeric constant in the code must trace to either (a) a design-doc citation, (b) a derivation from data properties, or (c) a comment explaining why this exact value. `0.01`, `10.0`, `1e-3` floating in code without justification are bugs waiting to happen.

## Research context

This package investigates the connection between vibrational systems as described in Rayleigh's Theory of Sound and developed in these previous pieces of work:
- Paper on https://arxiv.org/abs/2606.21535
- VDT: https://zenodo.org/records/20816835
- https://github.com/tuned-org-uk/spectral-transport-diff

### Driver of research
- Feature-space Laplacian components as directive for the diffusion process via spectral filters
- Optimization as recursive semantic search
- Dual-space hypothesis of graph wiring: every vector space is a geometric item-space and a semantic feature-space as in https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2

## Operational notes
- Always answer in English
- This environment uses bash
- When developing more than one file, write and push them one by one if needed
- For formulas in markdown files always use the `$$` syntax
- Use 3407 as random seed
- When using Python always use uv with the environment in the `.venv/`
- When using git use the profile github.com-coderbot to use the right SSH user, to add and commit use the pre-configured ssh host github.com-mecis with username Mec-iS and email tunedconsulting@gmail.com. Set the user.name and user.email at commit time using `git -c "user.name=Mec-iS" -c "user.email=tunedconsulting@gmail.com"` (do not use git config --global)
- Run tests with: `.venv/bin/python -m pytest tests/ -q`
- Run lint with: `.venv/bin/python -m ruff check esdm/ tests/`

## Architecture

The ESDM unifies two research tracks under one package:

### 1. Entropic-clock ESDM (ported from `esdm_legacy/`, now in `esdm/models/`)

The core model from `docs/00-DESIGN.md`. Replaces every external time parameter in VDT/SDM with an entropic time derived from spectral entropy exchange between latent (dark) and active (bright) sectors.

**Two-sector partition** (Barontini 2026 analogue):

$$\hat{H}_\mathit{sem} = \hat{H}_\mathit{item} + \hat{H}_\mathit{spectral} + \hat{H}_\mathit{wiring}(V)$$

- **Item sector** (bright): active spectral modes $\omega_k > \theta$, the `N_active` population
- **Spectral sector** (dark): latent wiring $L(z)$, inactive modes, ArrowSpace eigenpairs
- **Coupling** $V$: the `kl_S` regularisation weight (barrier height analogue)

**Entropic time**:

$$\tau_n = \sum_{i=1}^{n} |\Delta \mathit{kl\_tau}_i|$$

Monotonic by construction. Stops when entropy is not flowing (heat death). Computed entirely from `kl_tau` (the bright-sector KL), no external clock.

**Wave recurrence** (gated by entropic pump $\Lambda(\tau)$):

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, L_f Q_t - \gamma \Delta Q + \Lambda(\tau) \cdot \Delta\tau^2 \, B_t$$

**ELBO**:

$$\mathcal{L}_\mathit{ESDM} = \mathit{reconstruction} - \mathit{kl\_z} - \mathit{kl\_S} - \mathit{kl\_tau} - \beta \cdot |S_\mathit{bright} + S_\mathit{dark} - S_0|^2$$

### 2. 2.5-D dual-space score track (in `esdm/models/score3d.py`, `score25d.py`)

Score-based diffusion on minimal surfaces. The treatment condition under test: can a 2.5-D dual-space model (2D item coordinate + graph-Laplacian feature signal) match full 3D volumetric diffusion without computing Jacobian determinants?

**Target architecture (Phase 2)**: the entropic clock gates the 2.5-D score model. The fixed cosine schedule is replaced by $\bar{\alpha}(\tau)$. This is the design doc's actual claim (§5.2).

## Package layout

```
esdm/
  _vendor/              # VDT upstream, verbatim (commit 55cec39, relative imports)
    laplacian.py        # DifferentiableLaplacian, MassMatrix
    spectral.py         # TauModeDiffusion, tau_mode_kl, spectral_basis_kl, ...
    density.py          # SignedDensityMatrix (rho = rho+ - rho-)
    vdt.py              # VibrationalStateBlock (VDT), VDT
    wiring_decoder.py   # SpectralLoadingDecoder, WiringDecoder
    diffusion_decoder.py# DiffusionDecoder (SDM reverse chain)
  models/
    clock.py            # EntropicClock: tau_n, Lambda_tau, alpha_bar(tau)
    vibrational.py      # ESDM VibrationalStateBlock (wraps VDT, entropic dt)
    spectral_decoder.py # re-export shim for SpectralLoadingDecoder
    diffusion_decoder.py# EntropicDiffusionDecoder (alpha_bar from clock)
    elbo.py             # esdm_elbo (4-term + entropy conservation)
    spectral_kl.py      # tau_mode_kl_per_mode (de-duped from vendor)
    esdm.py             # ESDM top-level model (assembles all components)
    score3d.py          # 2.5-D track: 3D score network + DSM loss
    score25d.py         # 2.5-D track: dual-space score + graph diffusion step
  graphs/
    wiring.py           # WireGraph: arrowspace adapter (build, laplacian, rayleigh)
    laplacian.py        # DualSpaceMatrix: fused item-space matrix M_N
  data/
    minimal_surfaces.py # Synthetic Enneper/helicoid/catenoid generators
  sampling/
    euler_maruyama.py   # Reverse-SDE sampler (3D and 2.5-D)
  eval/
    metrics.py          # Chamfer distance, Rayleigh energy gap
    harness.py          # Comparison harness (3D vs 2.5-D)
  experiments/
    exp1.py             # Clock calibration (2-community synthetic graph)
    exp2.py             # Wave recurrence + CFL (path graph)
    exp3.py             # Barrier-height sweep (R^64 Gaussian clusters)
  cli/
    train.py            # esdm-train
    sweep.py            # esdm-sweep --kl-s-values ...
    build_laplacian.py  # esdm-build --features V.pt --output L_F.pt
  _configs/
    default.yaml        # Default model/clock/elbo/training config
```

## Primitives

### Graph construction
- **`WireGraph`** (`esdm.graphs.wiring`): the single surface through which all graph/Laplacian/Rayleigh-quotient computation flows. Wraps the `arrowspace` Rust bindings (with graceful fallback to a local k-NN Laplacian when the bindings are absent). API: `build(features, k, seed)`, `laplacian(normalized, sparse)`, `rayleigh_quotient(f)`, `spectral_embedding(n_components)`.
- **`DualSpaceMatrix`** (`esdm.graphs.laplacian`): fused item-space matrix $M_N = \alpha\|VV^\top\|_F - \beta\|V L_F V^\top\|_F$ from corpus embeddings. Consumes a Laplacian (does not build one) — complementary to `WireGraph`, not redundant. Chains as: `L_F = WireGraph.build(V.T()).laplacian()` → `DualSpaceMatrix().build(V, L_F)`.
- **`DifferentiableLaplacian`** (`esdm._vendor.laplacian`): trainable in-graph k-NN Laplacian builder (VDT upstream). Used by the vibrational block for CFL bounds.

### Entropic clock
- **`EntropicClock`** (`esdm.models.clock`): stateful module accumulating $\tau_n = \sum|\Delta \mathit{kl\_tau}|$, computing the entropic pump $\Lambda(\tau) = 1/|\partial_\phi \mathit{kl\_tau}|$, and emitting the cosine schedule $\bar{\alpha}(\tau) = \cos^2(\pi/2 \cdot \tau/\tau_\mathit{max})$. Detects heat death when $\tau_n$ stalls for > `heat_death_patience` steps.

### Spectral KLs
- **`tau_mode_kl`** (`esdm._vendor.spectral`): vendored scalar KL (sum over modes, mean over batch)
- **`tau_mode_kl_per_mode`** (`esdm.models.spectral_kl`): ESDM-layer unreduced $(B, q)$ variant for bright/dark sector splitting. Regression-tested against the vendored version (`test_spectral_kl_matches_vendor`).

## ArrowSpace role

ArrowSpace is the dual-space hypothesis: every vector space is simultaneously a geometric item-space and a semantic feature-space. In the ESDM partition:
- **Item-space** = bright sector (observed, reconstructed via $\hat{x}$)
- **Feature-space** = dark sector (Laplacian eigenpairs, spectral loading, ArrowSpace wiring $L_F$)

The `WireGraph` adapter (`esdm.graphs.wiring`) is the only place that imports `arrowspace`. It provides:
- Graph construction (k-NN + RBF, or via `ArrowSpaceBuilder` when available)
- Sparse/dense Laplacian (normalized and unnormalized)
- Rayleigh quotient $R(f) = f^\top L f / f^\top f$ (the $\lambda\tau$ / Dirichlet-energy proxy)
- Spectral embedding (leading non-trivial eigenvectors)

When `arrowspace` Rust bindings are absent, `WireGraph` falls back to a local k-NN Laplacian implementation (marked `# TEMPORARY: upstream issue`).

## Fixes applied during port from esdm_legacy

Three defects in the legacy `model.py` were fixed during the migration:

1. **L_z piping** (`esdm/models/esdm.py:forward`): the spectral decoder's $L_z$ is now fed to the vibrational block (was: identity `L_base`, leaving $L_z$ unused). Falls back to `L_base` when shape mismatch prevents direct piping.

2. **tau_mode_kl dedup** (`esdm/models/spectral_kl.py`): the inline per-mode KL formula is replaced by a call to `tau_mode_kl_per_mode`. Single source of truth; regression-tested against the vendored `tau_mode_kl`.

3. **kl_z** (`esdm/models/esdm.py:_kl_z`): the encoder now produces $(\mu, \mathrm{logvar})$ and `kl_z` is the standard Gaussian VAE KL $\frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$ (was: hard-coded to zero).

## Experiment design

- **Exp 1** (`esdm/experiments/exp1.py`): clock calibration on a 2-community synthetic graph. Pure clock unit test — `kl_tau` is scripted, not learned.
- **Exp 2** (`esdm/experiments/exp2.py`): wave recurrence + CFL on a 1-D path graph (n=32). Tests heat death detection and density matrix PSD.
- **Exp 3** (`esdm/experiments/exp3.py`): barrier-height sweep on 4 isotropic Gaussian clusters in $\mathbb{R}^{64}$. Sweeps `kl_S_weight`, `kl_decay`, `rho`, `lambda_clip`, seeds. First experiment to use the full clock+vibrational stack.

All experiments are synthetic. **Phase 3**: image/sound data loaders and experiments (pending).

## Test commands

```bash
# Full test suite
.venv/bin/python -m pytest tests/ -q

# Lint
.venv/bin/python -m ruff check esdm/ tests/

# Run a single experiment
.venv/bin/python -m pytest tests/test_exp3.py -q

# CLI smoke test
.venv/bin/python -m pytest tests/test_cli.py -q
```

## Autoresearch loop

The autoresearch loop is an autonomous iteration cycle that modifies model/experiment parameters, measures a composite fitness score, and keeps or discards each change. It runs the ESDM model on synthetic data, scores the run against the design doc's health criteria (§8.1), and iterates to maximize the score.

### Fitness metric

The composite fitness score (higher is better; criterion weights sum to 123) is defined in `esdm/eval/fitness.py` and computed from a single `run_single` call in `esdm/experiments/exp3.py`. It measures health criteria from docs/00-DESIGN.md §8.1. See `docs/03-FITNESS-CRITERIA.md` for the full design-to-fitness mapping and improvement pathways:

| Criterion | Max pts | What it measures |
|---|---|---|
| `tau_monotone` | 10 | $$\tau_n$$ strictly increasing |
| `tau_range_bonus` | 5 | Total $$\tau_n$$ accumulation (larger range = better) |
| `alpha_bar_decrease` | 10 | $$\bar{\alpha}(\tau)$$ drops from ~1 toward ~0 |
| `conservation` | 15 | $$\|S_\mathit{total} - S_0\| / S_0 \to 0$$ |
| `heat_death` | 10 | Heat death fires within the run |
| `heat_death_timing` | 5 | Fires around 70% through (not too early/late) |
| `rho_sign_stabilizing` | 10 | $$\rho_-/\rho_+$$ variance decreasing |
| `dark_entropy_floor` | 10 | Floor in [35%, 85%] of $$S_0$$ |

### Running the loop

```bash
# Establish baseline (iteration 0) -- current kept baseline
.venv/bin/python scripts/verify_fitness.py
```

### Loop configuration

| Parameter | Value |
|---|---|
| Goal | Maximize ESDM composite fitness score |
| Scope | `esdm/experiments/`, `esdm/models/`, `esdm/eval/fitness.py`, `scripts/verify_fitness.py` |
| Metric | Fitness score (higher is better) |
| Verify | `.venv/bin/python scripts/verify_fitness.py` |
| Guard | `.venv/bin/python -m pytest tests/ -q --tb=no` (all tests must pass) |
| Direction | `higher_is_better` |
| Iterations | 25 (bounded) |
| Results | `autoresearch/loop-{YYMMDD}-{HHMM}/results.tsv` |

### Decision protocol

Each iteration:
1. **Review** -- read the results TSV and `git log` to see what worked/failed
2. **Modify** -- one focused atomic change to a parameter, success criterion, or model component
3. **Commit** -- `experiment: {description}` prefix
4. **Verify** -- run `scripts/verify_fitness.py`, extract numeric score
5. **Guard** -- run full test suite; if any test fails, revert regardless of score
6. **Decide**:
   - `keep` -- score improved, guard passed
   - `discard` -- score worsened or unchanged -- `git revert HEAD --no-edit`
   - `crash` -- verify/guard failed -- `git revert HEAD --no-edit`
7. **Log** -- append row to `results.tsv`

### Test factoring for fast loops

When the loop metric is a test count (e.g., issue #21), the verify command must finish in seconds, not minutes. Follow these rules:

1. **Mark slow tests optional.** Use `@pytest.mark.slow` for any test that downloads data, loads a large frozen model, or encodes a dataset. Register the marker in `tests/conftest.py` and skip slow tests by default; run them on demand with `--run-slow` or `ESDM_RUN_SLOW=1`.
2. **Consolidate redundant assertions.** One test per behavior. Do not split shape, dtype, and normalisation into separate tests if they all exercise the same loader property.
3. **Test state, not interactions.** Assert on output tensors (shape, dtype, range, normalisation) rather than which private methods were called.
4. **Avoid data-dependent thresholds.** Do not assert on exact variance-retention fractions or dataset statistics unless the value is part of the contract.
5. **Keep helpers pure and fast.** Test preprocessing helpers with synthetic NumPy arrays; do not invoke downloads or model inference.
6. **Separate test sizes.** Use the `small/medium/large` model: small = pure logic, medium = local downloads/cache, large = external model inference.

Example verify/guard pair for a test-count loop:

```bash
# Fast verify (autoresearch loop)
.venv/bin/python -m pytest tests/test_data_loaders.py -q --tb=no

# Full guard (all tests, slow ones skipped by default)
.venv/bin/python -m pytest tests/ -q --tb=no

# Optional slow tests
.venv/bin/python -m pytest tests/test_data_loaders.py -q --tb=no --run-slow
```

### Baseline conditions

The current kept baseline uses the moderate-barrier condition with an extended run length:

| Parameter | Value | Rationale |
|---|---|---|
| `kl_S_weight` | 0.5 | Moderate barrier -- $$\tau_n$$ accumulates gradually, $$\bar{\alpha}$$ drops from ~1 to ~0.5 |
| `kl_decay` | 0.03 | Slow decay -- signal persists, entropic time spreads across the run |
| `lambda_clip` | 10.0 | Standard pump ceiling |
| `rho` | 3.0 | Primary SNR condition |
| `n_steps` | 800 | Extended run so heat death fires around the patience window |
| `seed` | 3407 | Project-wide seed |

The historical control used `n_steps=300` and scored 55.00; extending to 800 steps raised it to 68.58.

### Known weak spots (as of current baseline)

| Criterion | Score | Gap | Diagnosis |
|---|---|---|---|
| `conservation` | 1.71/15 | 13.29 | `conserved_ratio=0.89` — Lambda forcing dominates H contraction. The two-sector partition is now active (median threshold) but the energy balance is off. |
| `heat_death_timing` | 2.01/5 | 2.99 | Heat death fires at 92% of the run; target is 70%. |
| `net_energy_flow` | 0.00/5 | 5.00 | Lambda pump dominates H contraction throughout. Balance needs tuning. |

### Current baseline result

Latest baseline (code commit `4ba15ed`, CIFAR-10 DINO-256, 800 steps, lr=0.05):

```bash
.venv/bin/python scripts/verify_fitness.py --dataset cifar10 --n-steps 800 --lr 0.05
# 62.33
```

Fitness breakdown (max 85, real-data fitness):

| Criterion | Score | Max | Status |
|---|---|---|---|
| `tau_monotone` | 10.00 | 10 | OK |
| `tau_range_bonus` | 5.00 | 5 | OK |
| `alpha_bar_decrease` | 10.00 | 10 | OK |
| `conservation` | 1.71 | 15 | Gap: conserved_ratio=0.89, Lambda/H imbalance |
| `heat_death` | 10.00 | 10 | OK (fires at step 733/800) |
| `heat_death_timing` | 2.01 | 5 | Fires at 92%, target 70% |
| `rho_sign_stabilizing` | 10.00 | 10 | OK |
| `recon_decrease` | 8.61 | 10 | Near full (recon improves) |
| `net_energy_flow` | 0.00 | 5 | Lambda dominates H |
| `pi_contraction_ratio` | 5.00 | 5 | OK (passes [0.01, 0.3]) |
| **Total** | **62.33** | **85** | **73.3%** |

### Issue completion status

| Issue | Status | Notes |
|---|---|---|
| #20 — Complete entropic-clock track | **Complete** | Hybridisation operator H implemented (Phi+Diffuse+Pi). Fitness 62.33/85. Frozen-vs-learned ablation documented (delta +4.51). Dataset-agnostic plumbing fixed (median threshold, derived tau_max_est, derived eta from L_F). |
| #21 — Real data loaders | **Complete** | MNIST PCA-16 and CIFAR-10 DINO-256 loaders implemented, wired into CLI, tested. Audio deferred to #25. |
| #22 — Full training on real embeddings | **In progress** | Exp 4/5 training loops wired; CIFAR-10 baseline (lr=0.05, 800 steps on 2k-train subset) run. `scripts/verify_fitness.py` supports `--dataset {mnist,cifar10}`. Latent evaluation still pending. |
| #19 — Physics-based conservation | **Addressed** | Conservation gap reduced from 1.73 pts (synthetic) to current real-data gap. Root cause was missing H operator; now implemented. Remaining drift from Lambda/H balance. |
| #27 — Design validation vs Un-0 | **Complete** | Hybridisation operator (`docs/00-DESIGN_HYBRID.md`), phase-aware fitness criteria, diversity/quality metrics, Experiments 7–9 added to `docs/002-EXPERIMENTS.md`. |
| #23 — Generation and sampling | Pending | Blocked on #22. |
| #24 — 2.5-D dual-space track | Pending | Blocked on #20 (now unblocked). |
| #25 — Audio generation | Pending | Blocked on #21 (audio loader) and #23. |

### Architecture constants (from docs)

The following values are **constants**, not hyperparameters. They are set by the design docs and should not be swept in the autoresearch loop.

| Constant | Value | Source |
|---|---|---|
| $q$ (spectral modes) | $F/4$ | `docs/00-DESIGN_HYBRID.md` §5 |
| $\gamma$ (damping) | 0.1 | VDT default |
| `n_vibrational_steps` | 2 | CFL-safe for real $L_F$ (doc says 4, but 4 is unstable with ArrowSpace eigenvalues) |
| `lambda_clip` | 100.0 | `docs/00-DESIGN.md` §4.2 |
| `grad_clip` | 1.0 | Training stability |
| `seed` | 3407 | Project-wide rule |
| `batch_size` | 64 | Standard |
| `k` (ArrowSpace clusters) | 4 | Graph construction; tuning via `arrowspace_tuner` |

The only free hyperparameters are `lr` and `n_steps` (governed by heat death).

### Feature-space Laplacian

$L_F$ is built via **ArrowSpace** (`esdm/graphs/wiring.py`), not a custom k-NN fallback. The ArrowSpace Rust bindings return a sparse CSR or dense numpy array of shape $(F, F)$ — the feature-space Laplacian. This is loaded into PyTorch as `float32`.

- `WireGraph.build(data, k, seed, mode="feature")` — ArrowSpace feature-space Laplacian $(F, F)$
- `WireGraph.build(data, k, seed, mode="item")` — k-NN item-space Laplacian $(N, N)$ (2.5-D track only)
- Graph parameter tuning is deferred to `arrowspace_tuner` (not installed yet).

### Clock scaling

The entropic clock normalizes `kl_tau` by $S_0$ before accumulation: $\tau_n = \sum |\Delta (\mathit{kl\_\tau} / S_0)|$. This makes the clock scale-invariant with respect to the feature-space Laplacian's eigenvalue scale. $S_0$ is the total entropy budget from the first forward pass.

`tau_max_est` is initialized to $10 \cdot S_0$ so that $\bar{\alpha}(\tau) = \cos^2(\pi/2 \cdot \tau / \tau_\mathit{max})$ starts near 1 and decreases as entropy flows.

### No identity blending

$L_F$ is the real ArrowSpace graph Laplacian. It is **not** blended with identity. The noise/stochasticity in the system comes from the entropic clock via $\Lambda(\tau) \cdot \Delta\tau^2 \cdot B_t$, not from diluting $L_F$.

### Macro-phase roadmap to image and sound generation

The following macro-issues track the path from the current synthetic-diagnostic state to a model that produces images and sounds:

| Phase | Issue | Goal | Depends on |
|---|---|---|---|
| A — Complete entropic-clock track | [#20](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/20) | Full ESDM forward pass, fitness gaps closed | #19 |
| B — Real data loaders | [#21](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/21) | MNIST PCA-16, CIFAR-10 DINO-256, audio embeddings | independent |
| C — Full training on real embeddings | [#22](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/22) | Exp 4 (MNIST) + Exp 5 (CIFAR-10), latent evaluation | A, B |
| D — Generation and sampling | [#23](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/23) | Reverse-SDE sampler, pixel/audio decoder, FID/FAD evaluation | C |
| E — 2.5-D dual-space track | [#24](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/24) | score3d vs score25d on minimal surfaces, ablation, spectral stability | A |
| F — Audio generation | [#25](https://github.com/tuned-org-uk/entropic-semantic-diffusion/issues/25) | Audio data loader, waveform/spectrogram diffusion, audible samples | B, D |

### Next phases

1. **Close the conservation gap** (1.71/15) — the remaining drift comes from Lambda/H imbalance (`net_energy_flow` not near 0). Requires tuning the Lambda vs H equilibrium, not adding parameters.
2. **Tune heat-death timing** to ~70% (currently 92%). Adjust `heat_death_patience` or the relative threshold.
3. **Improve net_energy_flow** (0/5) — the Lambda pump dominates the H contraction. The balance may require a lower lr or longer warmup.
4. **Latent evaluation** (#22) — reconstruction quality, downstream k-NN accuracy, FID/FAD metrics.
5. **Generation pipeline** (#23) — reverse-SDE sampler, pixel decoder, FID evaluation.



## Autoresearch loop — issue #22

Issue #22 moves from synthetic diagnostics to full training on real embeddings. The loop starts with a test-count metric while scaffolding is built, then switches to a real-data fitness metric once training converges.

### Goal

Deliver issue #22: full training on real embeddings — Experiment 4 (MNIST PCA-16) and Experiment 5 (CIFAR-10 DINO-256), plus latent-space evaluation.

### Scope

- `esdm/experiments/exp4.py` — MNIST training experiment
- `esdm/experiments/exp5.py` — CIFAR-10 training experiment
- `esdm/eval/latent.py` — latent-space evaluation (reconstruction, conservation, downstream k-NN accuracy)
- `esdm/cli/train.py` — real-data training loop refinements
- `tests/test_exp4.py`, `tests/test_exp5.py`, `tests/test_latent.py` — fast + slow tests

### Loop configuration

| Parameter | Phase 1 (scaffolding) | Phase 2 (training) |
|---|---|---|
| Goal | Build and test Exp 4/5 + latent eval | Maximize real-data fitness (MNIST/CIFAR-10) |
| Metric | Count of passing tests in `tests/test_exp4.py` and `tests/test_exp5.py` | Composite fitness on real data (reconstruction + conservation + latent k-NN) |
| Direction | `higher_is_better` | `higher_is_better` |
| Verify | `.venv/bin/python -m pytest tests/test_exp4.py tests/test_exp5.py -q --tb=no` | `.venv/bin/python scripts/verify_fitness.py --dataset {mnist,cifar10}` |
| Guard | `.venv/bin/python -m pytest tests/ -q --tb=no` | `.venv/bin/python -m pytest tests/ -q --tb=no` |
| Iterations | 25 (bounded) | 25 (bounded) |
| Results | `autoresearch/loop-{YYMMDD}-{HHMM}/results.tsv` | `autoresearch/loop-{YYMMDD}-{HHMM}/results.tsv` |

### Decision protocol

Same as the generic autoresearch loop, with one addition: real-data slow tests (full MNIST PCA fit, CIFAR-10 DINO encode) are optional and run only with `--run-slow` or `ESDM_RUN_SLOW=1`. The fast verify command must finish in seconds.

### Initial todos

See `autoresearch/loop-260714-1458/todos.md`.
