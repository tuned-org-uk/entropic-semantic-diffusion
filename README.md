# Entropic Semantic Diffusion (ESD)

> A learning architecture in which time is not given — it is *earned* from entropy.

> **Note:** the repository slug retains the historical spelling `enthropic-semantic-diffusion`; the Python package name is `esdm`.

---

## Background: The ArrowSpace Dual-Space Hypothesis

Modern vector retrieval ranks items by cosine similarity or inner product in an ambient embedding space. This geometric signal, however, does not account for the statistical and topological structure of the corpus from which embeddings are drawn. Items may be geometrically close while belonging to structurally different regions of the feature manifold; geometrically distant items may share deep structural affinity through corpus topology. This is the **semantic gap** — a corpus-level signal invisible to pairwise geometric distance.

**ArrowSpace** ([tuned.org.uk](https://www.tuned.org.uk), [Mec-iS/arrowspace-concept-paper](https://github.com/Mec-iS/arrowspace-concept-paper)) addresses this gap with a foundational dual-space hypothesis:

> *Every embedding space is simultaneously a geometric **item-space** and a semantic **feature-space**. A corpus of $N$ items embedded in $F$ dimensions is not just a cloud of points in $\mathbb{R}^F$ — it is also a graph whose nodes are the $F$ features and whose edge weights are determined by item co-occurrence structure. The Laplacian $L_F$ of this feature-space graph encodes topological information about semantic relationships that cannot be recovered from pairwise distances alone.*

The feature-space graph Laplacian $L_F$ has eigenpairs $(\lambda_k, u_k)$ that decompose the corpus into vibrational modes. The Fiedler vector $u_2$ (eigenvector of the smallest nonzero eigenvalue $\lambda_2$) defines the primary axis of semantic variation — it identifies the **deepest structural split** in the corpus, analogous to a standing wave's fundamental mode on a vibrating string.

This spectral structure is *information-theoretically certified* by **epiplexity**: a two-part Minimum Description Length criterion that proves the Laplacian $L_F$ encodes structural information beyond the original geometric scores, separating learnable regularity (structural bits $S_T$) from irreducible per-sample uncertainty (random bits $H_T$).

### Item-space vs. Feature-space

| Dimension | Item-space | Feature-space |
|---|---|---|
| Nodes | Corpus items ($N$) | Embedding dimensions ($F$) |
| Edges | Geometric similarity | Feature co-occurrence / spectral loading |
| Laplacian | $L_I \in \mathbb{R}^{N \times N}$ | $L_F \in \mathbb{R}^{F \times F}$ |
| Eigenvectors | Cluster structure over items | Vibrational modes of the semantic manifold |
| Role in ESD | Bright sector (observed) | Dark sector (latent, governs dynamics) |

---

## Motivation: Barontini's Experiment (2026)

> Barontini, G. *Testing the problem of time with cold atoms*. Phys. Rev. Research **8**, L022047 (2026). DOI: [10.1103/1h9j-df4k](https://doi.org/10.1103/1h9j-df4k)

The Wheeler-DeWitt equation $\hat{H}\Psi = 0$ — the master equation of canonical quantum gravity — admits no external time parameter. This is the **problem of time**: a fundamentally closed system has no built-in mechanism to sequence its own evolution.

Barontini realises an analogue of this problem in a controlled laboratory: a Bose-Einstein condensate of $\approx 24\,000$ ${}^{87}$Rb atoms oscillating in a conservative trap partitioned by a thin optical barrier. One side of the barrier is observed (**bright sector**); the other is unobserved (**dark sector**). The total system is closed and governed by a time-independent Hamiltonian. The question asked is: *can the dynamics of the bright sector be ordered using only quantities internal to it, without reference to the external laboratory clock?*

The answer is **yes**, via **entropic time**:

$$\tau(\lambda) = \frac{\sigma}{k_B} \int_\lambda \frac{dS}{d\phi} \, |d\phi|$$

where $S$ is the coarse-grained entropy of the bright sector, $\phi$ is the clock field (centre-of-mass coordinate), and $\sigma$ is an arbitrary time unit. This is monotonically increasing by construction — the arrow of time emerges from entropy exchange, not from an external clock.

Three key experimental results:

1. **$\tau$ is a robust ordering parameter.** Across different barrier heights $V$, $\tau$ grows monotonically and correctly orders all bright-sector events from "big bang" (first atom entry) to "big crunch" (last atom departure).
2. **Entropy is conserved globally.** At every barrier height, $S_\text{bright}(\tau) + S_\text{dark}(\tau) \approx S_0$ — the total miniuniverse entropy is constant even as entropy flows between sectors.
3. **An entropic Schrödinger equation reproduces the data.** Replacing external time $t$ with $\tau$ in the effective Hamiltonian (via Feshbach-Villars decomposition of the static WDW equation) yields:

$$i\hbar \,\partial_\tau \psi(\tau, a) = \Phi(\tau)\psi(\tau, a) + \Lambda(\tau) \, H_\text{geom} \, \psi(\tau, a)$$

where $\Lambda(\tau) = (\partial_\phi S)^{-1} k_B / \sigma\omega\phi$ is the **entropic pump** — an entropy-dependent modulator of the geometric Hamiltonian. Its numerical solutions quantitatively match the measured BEC dynamics.

---

## The ESD Architecture

The Entropic Semantic Diffusion model is a closed two-sector learning system whose dynamics are parametrised entirely by internal entropy exchange — **no external epoch counter, no fixed schedule, no external clock**.

### The Wheeler-DeWitt Problem in Machine Learning

Prior architectures in this research programme share a structural flaw: time is external.

- **VDT** ([vibrational-deduction-transformer](https://github.com/tuned-org-uk/vibrational-deduction-transformer)): wave recurrence index `t` is an integer step counter with no thermodynamic meaning.
- **SDM** ([spectral-diffusion](https://github.com/tuned-org-uk/spectral-diffusion)): `cosine_alpha_bar(t, T=200)` is a fixed schedule blind to what the spectral system is doing.

The ESDM replaces both external clocks with entropic time accumulated from the spectral entropy flow of the semantic system itself.

### Two-Sector Partition

The mapping from Barontini's physical experiment to the semantic architecture:

| Barontini (BEC) | ESDM (semantic) |
|---|---|
| Bright sector (observed half) | Active spectral modes: $\omega_k > \theta$ — `N_active` |
| Dark sector (unobserved half) | Latent $L(z)$, inactive modes, ArrowSpace eigenpairs |
| Coupling $\hat{H}_\text{coupling}$ | `kl_S` regularisation weight |
| Barrier height $V$ | `kl_S` sweep parameter |
| Clock field $\phi$ (centre of mass) | `log_lam_s` — learnable log spectral scale |
| Scale factor $a$ (BEC RMS radius) | Reconstruction width $\Sigma$ |
| Big bang / big crunch | Mode activation / mode collapse events |
| Heat death | `N_active` plateau — `tau_n` stalls |

### Entropic Time

The discrete entropic time accumulator, analogous to Barontini's Eq. (3):

$$\tau_n = \sum_{i=1}^{n} |\Delta \, \mathit{kl\_tau}_i|$$

This is the sum of absolute changes in the tau-mode KL term of the VDT ELBO. It is **monotonically increasing by construction**. When `kl_tau` is stable (no entropy exchange between sectors), $\tau_n$ stops — no time elapses, consistent with Barontini's observation that no entropic time passes between a big crunch and the subsequent big bang.

### The Entropic Pump

The attention-forcing term $B_t$ in the VDT wave recurrence is gated by $\Lambda(\tau)$, the discrete entropic pump:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, L_f Q_t - \gamma \Delta Q + \Lambda(\tau) \cdot \Delta\tau^2 \, B_t$$

where $$\Lambda(\tau) = |\partial_\phi \mathit{kl\_tau}|^{-1}$$. When modes are stable (small entropy derivative), $\Lambda$ is large and attention drives exploration. When modes are in rapid transition (large entropy derivative), $\Lambda$ is small and the wave recurrence runs as free vibrational dynamics.

### Entropic Diffusion Schedule

The SDM's fixed cosine schedule is replaced by one indexed by accumulated entropic time:

$$\bar{\alpha}(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau}{\tau_\text{max}}\right)$$

Steps where `kl_tau` changes rapidly advance the diffusion schedule faster — dissolving semantic structure at precisely the moments the spectral geometry is reorganising. Steps at plateau contribute negligibly. The schedule adapts to the corpus; no $T$ needs to be preset.

### Architecture Pipeline

```
Input x  (B, D)
     │
     ▼
┌─────────────────────────┐
│  EntropicClock   [NEW]  │  accumulates τ from |Δkl_tau|
│                         │  computes Λ(τ), dark_sector_entropy
└────────────┬────────────┘
             │  τ, Λ(τ), Δτ
             ▼
┌─────────────────────────┐
│  VibrationalStateBlock  │  VDT wave recurrence
│       [MODIFIED]        │  gated by Λ(τ), dt → Δτ
└────────────┬────────────┘
             │  Q_t, ρ_+, ρ_-
             ▼
┌─────────────────────────┐
│  SpectralLoadingDecoder │  VDT — unchanged
│                         │  z, U_q → W, ω, N_active, log_lam_s
└────────────┬────────────┘
             │  L(z), N_active, log_lam_s → feeds back to EntropicClock
             ▼
┌─────────────────────────┐
│  EntropicDiffusion      │  SDM reverse chain
│  Decoder  [MODIFIED]    │  ᾱ(τ) replaces cosine_alpha_bar(t, T)
└────────────┬────────────┘
             │  x̂
             ▼
      Prediction / Generation
```

### The ESDM ELBO

The objective extends the VDT ELBO with an entropy conservation term:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(x \mid z, W)]}_{\mathit{reconstruction}} - \underbrace{\mathrm{KL}(q(z) \,\|\, \mathcal{N}(0,I))}_{\mathit{kl\_z}} - \underbrace{\mathrm{KL}(q(S) \,\|\, p(S|I))}_{\mathit{kl\_S}} - \underbrace{\mathrm{KL}(q(\omega) \,\|\, p(\omega|\tau,L))}_{\mathit{kl\_tau}} - \underbrace{\beta \cdot |S_\mathit{bright}(\tau) + S_\mathit{dark}(\tau) - S_0|^2}_{\mathit{entropy conservation}}$$

The conservation term enforces that entropy flowing from dark to bright sector is balanced globally — directly mirroring the BEC miniuniverse conservation law verified by Barontini.

### Signed Interference as Confidence

The VDT `SignedDensityMatrix` tracks $\rho = \rho_+ - \rho_-$. When $\rho_- > \rho_+$ (destructive interference dominates), the model is in a decoherent state structurally analogous to a big crunch — the entropic clock tends to stall. A confidence signal derived from this ratio:

$$\text{confidence}(\tau) = \frac{\rho_+}{\rho_+ + \rho_- + \epsilon}$$

can flag low-quality predictions without any external validation signal.

---

## Training Phases

Using Barontini's cosmological language, a healthy training run passes through five identifiable phases:

| Phase | Signal | Analogue |
|---|---|---|
| **Big bang** | `tau_n` starts, `N_active` rises | BEC atoms entering bright sector |
| **Expansion** | `conserved` stable, `Lambda_tau` moderate | BEC reaching maximum extent |
| **Maximum extension** | `N_active` plateau, $\bar{\alpha} \approx 0.5$ | BEC at turning point |
| **Contraction / big crunch** | `N_active` falls, `rho_sign` rises | BEC recollapsing |
| **Heat death** | `tau_n` stalls, `conserved` $\approx 0$ | BEC stationary — training ends |

---

## Repository Layout

```
entropic-semantic-diffusion/
├── README.md                    ← this file
├── docs/
│   ├── 00-DESIGN.md             ← full architecture design document
│   ├── 01-IMPLEMENTATION-PLAN.md
│   └── 02-EXPERIMENTS.md        ← experiment ladder (Exp 1–5)
├── esdm/                        ← Python package
│   ├── clock.py
│   ├── experiments/             ← standalone experiment scripts
│   │   ├── exp1.py
│   │   └── exp2.py
│   ├── laplacian.py
│   ├── model.py
│   └── ...
├── scripts/                     ← thin CLI wrappers
│   ├── run_exp1.py
│   ├── train.py
│   └── ...
├── tests/                       ← pytest suite
└── runs/                        ← experiment outputs (created on run)
```

### docs/00-DESIGN.md

The design document covers:

1. **The Core Problem** — why VDT/SDM's external time is the WDW problem of time
2. **Two-Sector Partition** — BEC dark/bright partition mapped onto latent/active spectral modes, with full correspondence table
3. **Entropic Time** — discrete accumulator, key properties, non-monotonicity fix
4. **Entropic Schrödinger Equation** — Barontini's Eq. 6 mapped to the gated VDT wave recurrence
5. **Entropic Diffusion Schedule** — replaces `cosine_alpha_bar(t, T)`, including per-mode variant
6. **Architecture** — four-module pipeline with full module specifications
7. **ESDM ELBO** — extended objective with entropy conservation term
8. **Training Diagnostics** — per-batch health monitoring loop and regime identification
9. **Barrier Height Sweep** — `kl_S` sweep protocol mirroring Barontini's Fig. 2
10. **Thermal Time and ArrowSpace** — connections to Connes-Rovelli, dual-space hypothesis, recursive semantic search
11. **Open Questions** — five concrete research questions
12. **Implementation Checklist** — ten actionable tasks

---

## Running the Experiments

The experiment ladder is described in `docs/02-EXPERIMENTS.md`.  Each experiment can be run from the repository root with `uv` or the project virtual environment.

### Experiment 1 — Entropic Clock Calibration

A pure unit test of `EntropicClock` on a synthetic two-community graph with a scripted four-phase KL signal:

```bash
.venv/bin/python scripts/run_exp1.py
```

This sweeps the bridge weight `w ∈ {0.0, 0.01, 0.1, 1.0}` and writes per-step metrics to:

```
runs/exp1/<seed>/w_<weight>/metrics.csv
```

The script asserts the success criteria from `docs/02-EXPERIMENTS.md` §Experiment 1:
- `tau_n` is flat during phases A and C,
- `tau_n` advances during phases B and D,
- `alpha_bar(tau / tau_max_est = 0.5) = 0.5`,
- `Lambda_tau` stays finite and clipped,
- no NaN/Inf in buffers, and no heat death within 250 steps.

Run a custom output directory or seed:

```bash
.venv/bin/python scripts/run_exp1.py --output-dir runs/exp1 --seed 42
```

### Experiment 2 — Vibrational Wave Recurrence on a 1-D Harmonic Chain

A unit test of `VibrationalStateBlock` on a 32-node path graph, driven by a live `EntropicClock` with decaying entropy flow:

```bash
.venv/bin/python scripts/run_exp2.py
```

Outputs are written to:

```
runs/exp2/<seed>/metrics.csv
```

The script asserts:
- the path graph Laplacian matches the analytic spectrum `λ_k = 2 - 2·cos(πk/32)`;
- the CFL-clamped time step `dt` never exceeds `sqrt(2 / λ_max)`;
- `Q_t[0]` stays finite and does not blow up over 500 steps;
- `is_heat_death()` fires by step 400 ± 50 (with the default decay);
- `rho_plus` and `rho_minus` remain finite and PSD;
- `dark_entropy` and `conserved` are logged for the entropy-balance diagnostic.

The default `kl_decay=0.08` consumes the entropy budget quickly and matches the
issue's heat-death timing.  The entropic-pump ceiling defaults to
`lambda_clip=15.0` so that the Phase-3 pump explosion does not immediately
saturate.  Slower decays spread the schedule but delay heat death.  The
bright/dark mode split drives the spectral dark-entropy proxy, and the CSV
also logs `g_mean` and `g_entropy` for the WHT-gate diagnostic convention:

```bash
.venv/bin/python scripts/run_exp2.py --output-dir runs/exp2 --seed 42
.venv/bin/python scripts/run_exp2.py --kl-decay 0.02 --clock-eps 1e-4
.venv/bin/python scripts/run_exp2.py --lambda-clip 10.0 --bright-modes 8
```


### Experiment 3 — Entropic Diffusion on the Fused Item Kernel

A wiring-level integration test of the full ESDM stack on a synthetic corpus,
combining the entropic clock, vibrational block, dual-space fused item kernel,
and entropic diffusion decoder:

```bash
.venv/bin/python scripts/run_exp3.py
```

Outputs are written to:

```
runs/exp3/<seed>/metrics.csv
```

The script asserts:
- the fused item-space kernel remains symmetric and numerically stable;
- `tau_n` advances monotonically and the entropic schedule stays within `[0, 1]`;
- `Lambda_tau`, `dark_entropy`, and `conserved` remain finite throughout the run;
- WHT-gate diagnostics (`g_mean`, `g_entropy`) are logged alongside the core clock signals;
- no NaN/Inf values appear in model states or metrics, and no premature heat death occurs under the default configuration.

Run a custom output directory or seed:

```bash
.venv/bin/python scripts/run_exp3.py --output-dir runs/exp3 --seed 42
```

### Test Suite

Run all unit and experiment tests:

```bash
.venv/bin/python -m pytest
```

Experiment 1 is also covered by `tests/test_exp1.py`.

### Autoresearch Loop

The repository includes a self-contained autoresearch driver (`scripts/autoresearch_loop.py`)
that iteratively tweaks ESDM hyperparameters, measures the composite fitness score, and
keeps only changes that improve it.  The loop is configured from `AGENTS.md`.

Run the loop with the project virtual environment:

```bash
.venv/bin/python scripts/autoresearch_loop.py
```

By default it performs 25 bounded iterations.  Override the iteration count or enable
mid-loop eval checkpoints:

```bash
.venv/bin/python scripts/autoresearch_loop.py --iterations 50
.venv/bin/python scripts/autoresearch_loop.py --evals --evals-interval 5
```

The driver writes a timestamped TSV to `autoresearch/loop-YYMMDD-HHMM/results.tsv` and a
`handoff.json` for downstream chaining.  It only commits to git when a change improves the
fitness score; unsuccessful changes are reverted automatically.

---

## Related Repositories

| Repo | Role |
|---|---|
| [tuned-org-uk/vibrational-deduction-transformer](https://github.com/tuned-org-uk/vibrational-deduction-transformer) | VDT — wave recurrence, ELBO, `VibrationalStateBlock` |
| [tuned-org-uk/quantum-vibrational](https://github.com/tuned-org-uk/quantum-vibrational) | Real amplitudes, signed interference, amplitude Laplacian $\Lambda_A$ |
| [tuned-org-uk/spectral-diffusion](https://github.com/tuned-org-uk/spectral-diffusion) | SDM — forward process, `cosine_alpha_bar`, LGMRF prior |
| [Mec-iS/arrowspace-concept-paper](https://github.com/Mec-iS/arrowspace-concept-paper) | ArrowSpace / SPIN — feature-space Laplacian, epiplexity, dual-space hypothesis |

---

## References

- Barontini, G. (2026). *Testing the problem of time with cold atoms*. Phys. Rev. Research 8, L022047. DOI: 10.1103/1h9j-df4k
- Connes, A. & Rovelli, C. (1994). *Von Neumann algebra automorphisms and time-thermodynamics relation*. Class. Quantum Grav. 11, 2899.
- DeWitt, B.S. (1967). *Quantum theory of gravity I: The canonical theory*. Phys. Rev. 160, 1113.
- Page, D.N. & Wootters, W.K. (1983). *Evolution without evolution*. Phys. Rev. D 27, 2885.
- Moriondo, L. (2025). *Vibrational Deductive Transformer*. github.com/tuned-org-uk/vibrational-deduction-transformer
- Moriondo, L. (2025). *Real Amplitudes and Signed Interference in Vibrational Systems*. github.com/tuned-org-uk/quantum-vibrational
- Moriondo, L. (2025). *Spectral Diffusion Model*. github.com/tuned-org-uk/spectral-diffusion
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*. [tuned.org.uk/posts/020_arrowspace_semantic_basins_part2](https://www.tuned.org.uk/posts/020_arrowspace_semantic_basins_part2)
