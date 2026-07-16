# 00-DESIGN_HYBRID — The Hybridisation Operator

> **Status:** Partially implemented — v0.2.0
> **Depends on:** `00-DESIGN.md` §4, `001-IMPLEMENTATION-PLAN.md` §4.2–4.3
> **Addresses:** the implicit mixing step between the F×F feature-space
> Laplacian signal and the N×F geometric item-space state that was
> embedded silently inside the wave recurrence in `00-DESIGN.md` §4.2/4.3
> but never written as a named, closed-form operator.
>
> **Implementation history:**
> - Issue #20 (2026-07-15): `_hybridisation()` fully implemented in
>   `esdm/models/vibrational.py`. WHT filter removed as redundant.
>   `eta` made a derived constant (not a free config parameter).
> - Issue #23 (2026-07-15): reverse-SDE generation pipeline added in
>   `esdm/sampling/entropic_sampler.py`. The reverse path uses the DDPM
>   ancestral update and does **not** call `_hybridisation()`.
> - Issue #29 (open): generation validation. Experiment 4 Condition C
>   (unsigned lifting ablation) is blocked on this issue.

---

## 1. The Problem

ESDM operates on two spaces simultaneously:

- **Feature space** (dark sector): $L_F \in \mathbb{R}^{F \times F}$ — the
  graph Laplacian of the *feature dimensions*, built by ArrowSpace graph
  wiring. Its leading eigenpairs $(U_q, \Lambda_q)$ encode the spectral
  geometry of the semantic field.
- **Item space** (bright sector): $Q_t \in \mathbb{R}^{B \times F}$ — one
  row per item in the batch, evolving through the string/spring wave
  recurrence. Each row is a point in $\mathbb{R}^F$ that represents an
  item's current vibrational state.

The term $\widetilde{L_F Q_t}$ in the wave recurrence mixes these two spaces:
$L_F$ (F×F, feature geometry) acts on $Q_t$ (B×F, item states), and the
result is fed back into the item-space dynamics. This is the **hybridisation
step** — the mechanism by which spectral structure in the dark sector directs
the diffusion of the bright sector.

In `00-DESIGN.md` §4.2–4.3, this step is written as:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2\,\widetilde{L_F Q_t}
  - \gamma \Delta Q + \Lambda(\tau)\cdot\Delta\tau^2\,B_t$$

but the **lifting** (how an item-state enters the feature space before $L_F$
acts) and the **projection** (how the Laplacian-diffused feature signal
returns to the item space) are left implicit. This document makes them
explicit.

---

## 2. The Hybridisation Operator: Three Steps

The full hybridisation is a three-step operator
$\mathcal{H}: \mathbb{R}^{B \times F} \to \mathbb{R}^{B \times F}$
applied at every wave-recurrence step:

$$\mathcal{H}(Q_t) = \Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)$$

### Step 1 — Lifting $\Phi$

$$\Phi:\; Q_t \in \mathbb{R}^{B \times F} \;\longrightarrow\; F_t \in \mathbb{R}^{B \times F}$$

$$\boxed{F_t = Q_t \, U_q \, U_q^\top}$$

where $U_q \in \mathbb{R}^{F \times q}$ are the $q$ leading non-trivial
eigenvectors of $L_F$ (the ArrowSpace spectral embedding), with
$q \leq F$.

**Interpretation.** $\Phi$ projects each item's vibrational state onto the
subspace spanned by the $q$ dominant feature-space modes, then lifts it
back to $\mathbb{R}^F$. The result $F_t$ is a **spectrally filtered
version of $Q_t$**: components orthogonal to the leading eigenvectors (i.e.
high-frequency, locally incoherent modes) are suppressed before the
Laplacian acts. This is the item-space analogue of the ArrowSpace taumode
projection — it restricts the spring/string system to the
semantically coherent subspace of feature space.

**Relation to existing architecture.** $U_q$ is already produced by
`SpectralLoadingDecoder` as part of the spectral loading prior
$(U_q, \Lambda_q)$. The lifting step reuses these eigenvectors without
introducing new parameters.

**Degenerate cases.**
- $q = F$: $U_q U_q^\top = I$, so $\Phi = \text{id}$ — no projection,
  full feature space passed through. This recovers the implicit assumption
  in `00-DESIGN.md` §4.2.
- $q = 1$: only the Fiedler vector is retained — maximal compression to
  the single most semantically coherent mode.

---

### Step 2 — Feature-space Laplacian diffusion

$$\text{Diffuse}_{L_F}:\; F_t \in \mathbb{R}^{B \times F}
  \;\longrightarrow\; \tilde{F}_t \in \mathbb{R}^{B \times F}$$

$$\boxed{\tilde{F}_t = F_t - \frac{\sigma^2 \Delta\tau}{2}\,L_F F_t}$$

where $L_F F_t$ is the feature-space Laplacian action.

**Interpretation.** Each item's lifted feature signal diffuses along the
F×F feature-space graph. Items whose feature signals are far from the
graph's smooth modes are pulled toward them; items already in a smooth
mode are held there.

**Note on $\sigma^2$ vs $\Delta\tau^2$.** The diffusion coefficient here
uses $\sigma^2 \Delta\tau / 2$ (a heat-equation step), not $\Delta\tau^2$
(a wave-equation step). This is deliberate: Step 2 is a **parabolic**
(irreversible) smoothing of the feature signal, while the wave recurrence
in Step 3 is **hyperbolic** (oscillatory). The hybridisation operator
combines both character types: diffusion in feature space, oscillation in
item space.

---

### Step 3 — Projection $\Pi$

$$\Pi:\; \tilde{F}_t \in \mathbb{R}^{B \times F}
  \;\longrightarrow\; \Delta Q_t \in \mathbb{R}^{B \times F}$$

$$\boxed{\Delta Q_t = -\eta \,\nabla_{Q_t} R(\tilde{F}_t)}$$

where the Rayleigh gradient is:

$$\nabla_{Q_t} R(\tilde{F}_t)
  = \frac{2}{{\|\tilde{F}_t\|^2}}\left(L_F \tilde{F}_t
  - R(\tilde{F}_t)\,\tilde{F}_t\right) U_q^\top$$

with $R(\tilde{F}_t) = \tilde{F}_t^\top L_F \tilde{F}_t / \tilde{F}_t^\top
\tilde{F}_t$ the Rayleigh quotient (ArrowSpace taumode score), computed
row-wise across the batch, and $\eta > 0$ a step size derived from the
spectral scale of $L_F$ (see §5).

**Sign convention in code.** `_hybridisation()` returns
`H_action = eta * grad_R` (analogous to `L*Q`). The wave recurrence
applies `- dt² * H_action`, giving the restoring force
$-\eta \Delta\tau^2 \nabla R$ toward low-energy modes. The design-doc
sign $\Delta Q_t = -\eta \nabla R$ and the code are consistent when read
together with the `-dt²` factor in `_step()`.

**Interpretation.** $\Pi$ converts the diffused feature signal $\tilde{F}_t$
back into a **gradient force** on the item-space state $Q_t$. Items whose
feature signals have high Rayleigh quotient (far from smooth modes,
high spectral energy) experience a larger restoring force pulling them
toward lower-energy regions of the feature-space graph. Items already in
a smooth mode experience near-zero force. This is precisely the
string/spring restoring force: the Rayleigh quotient measures how far a
string is stretched relative to the graph's natural modes, and
$-\nabla R$ is the force that relaxes it.

**Relation to $\nabla_z R$ in `001-IMPLEMENTATION-PLAN.md` §4.2.**
The 2.5-D implementation uses $\nabla_z R(f_t)$ where $z$ is a 2D item
coordinate. The closed-form expression above is the F-dimensional
generalisation: the $U_q^\top$ term at the end projects the F-dimensional
gradient back through the lifting subspace, ensuring $\Delta Q_t$ lies in
the same subspace as $F_t = Q_t U_q U_q^\top$.

---

## 3. Full Hybridisation in the Wave Recurrence

Substituting all three steps into the ESDM wave recurrence, the complete
update equation with explicit hybridisation is:

$$\boxed{
Q_{t+1} = 2Q_t - Q_{t-1}
\;-\; \Delta\tau^2\,\Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)
\;-\; \gamma \Delta Q
\;+\; \Lambda(\tau)\cdot\Delta\tau^2\,B_t
}$$

Expanding:

$$Q_{t+1} = 2Q_t - Q_{t-1}
\;+\; \eta\,\Delta\tau^2\,\nabla_{Q_t} R\!\left(
  Q_t U_q U_q^\top
  - \tfrac{\sigma^2\Delta\tau}{2}\,L_F (Q_t U_q U_q^\top)
\right)
\;-\; \gamma \Delta Q
\;+\; \Lambda(\tau)\cdot\Delta\tau^2\,B_t$$

This is the **canonical ESDM string/spring equation** with hybridisation
made explicit. The three operators act in sequence:

1. $\Phi$ — restrict to semantically coherent subspace (spectral projection)
2. $\text{Diffuse}_{L_F}$ — smooth along feature-space graph (Laplacian heat flow)
3. $\Pi$ — convert smoothed signal back to item-space restoring force (Rayleigh gradient)

---

## 4. Architectural Diagram

```
        Item space (bright sector)             Feature space (dark sector)
        Q_t  ∈  R^{B×F}                        L_F  ∈  R^{F×F}
              │                                        │
              │  Φ: project onto U_q subspace         │
              ▼                                        │
        F_t = Q_t U_q U_q^T  ∈  R^{B×F}              │
              │                                        │
              │  Diffuse: L_F heat step ◄─┘
              ▼
        F̃_t  ∈  R^{B×F}
              │
              │  Π: Rayleigh gradient → restoring force
              ▼
        ΔQ_t  ∈  R^{B×F}
              │
              ▼
        Q_{t+1} = wave recurrence(Q_t, ΔQ_t, Λ(τ)·B_t)

        [Reverse path: entropic_sampler.py uses DDPM ancestral update.
         H is NOT called during generation — see §7 Q3.]
```

---

## 5. Parameter Summary

| Symbol | Shape | Source | Learned? | Implementation status |
|---|---|---|---|---|
| $U_q$ | $F \times q$ | `SpectralLoadingDecoder` eigenvectors | No (buffer, not Parameter) | Frozen after graph build. Differentiable $U_q$ deferred to Phase 3. |
| $q$ | scalar | `dim_latent` in config; default $F/4$ rounded to power of 2 | No | `--q` CLI flag available in `verify_fitness.py` |
| $\sigma^2$ | scalar | `sigma_sq` in config; default `1.0` | Optional (`freeze_h=False`) | `nn.Parameter`, frozen when `freeze_h=True` |
| $\eta$ | scalar | **Derived:** `0.1 * F / (2 * λ_max(L_F))` at init | Optional (`freeze_h=False`) | **Not a free config parameter** — removed from config/CLI in #20 refactor. Computed from spectral scale to be dataset-agnostic. |
| $\Lambda(\tau)$ | scalar | `EntropicClock` output | No (derived) | Live output from `clock.Lambda_tau` |

---

## 6. Degenerate and Special Cases

| Condition | Effect |
|---|---|
| $q = F$ | $\Phi = \text{id}$, no projection → recovers the implicit `00-DESIGN.md` §4.2 formulation |
| $\sigma^2 \to 0$ | Diffuse step vanishes → $\tilde{F}_t = F_t$ → hybridisation reduces to pure Rayleigh-gradient restoring force on the projected state |
| $\eta \to 0$ | Projection step vanishes → feature-space diffusion has no effect on item dynamics → dark and bright sectors decouple |
| $q = 1$ | Only the Fiedler vector is used → maximum compression; item dynamics are governed entirely by the graph's primary connectivity axis |

---

## 7. Open Questions — Status Against v0.2.0

### Q1 — Backpropagation through $\Phi$ (differentiable $U_q$)

**Status: deferred to Phase 3. No change from original resolution.**

Issue \#27 identifies the core empirical risk as not knowing whether the
spectral machinery does real work *beyond a fixed basis*. Making $U_q$
differentiable would make the lifting step jointly optimisable — but it would
also conflate two separate claims: (a) the fixed ArrowSpace spectral geometry
contributes, and (b) an adapted lifting subspace contributes further. The
frozen-vs-learned ablation (Experiment 7, `002-EXPERIMENTS.md`) is designed
to isolate (a) first. Until that experiment is run and (a) is confirmed, a
differentiable $U_q$ is premature.

**Decision rule:**
- If Experiment 7 frozen-gate ≈ learned-gate: the spectral basis itself is
  not contributing beyond a scaling layer; differentiating $U_q$ is low
  priority.
- If learned-gate substantially outperforms frozen-gate: spectral frequency
  selection is real; differentiating $U_q$ becomes the natural Phase 3
  question — how much further gain comes from adapting the lifting subspace
  itself.

**Implementation note:** if $U_q$ is eventually made differentiable, the
correct approach is a Lanczos-based differentiable eigendecomposition
(e.g. `torch.lobpcg` with `largest=False`) rather than full
`torch.linalg.eigh`, since only the leading $q \ll F$ eigenvectors are
needed. Mark as `# TEMPORAL: upstream issue` per `001-IMPLEMENTATION-PLAN.md`
§3 rules until a native `arrowspace` differentiable eigenbasis lands.

---

### Q2 — Choice of $q$

**Status: default implemented ($q = F/4$); sweep CLI wired; sweep not yet run.**

Issue \#27 Gap 4 asks whether spectral diffusion improves *semantic basin
coverage* or merely sharpens samples within dominant modes. The choice of $q$
directly controls this:

| $q$ | Lifting behaviour | Predicted effect on diversity |
|---|---|---|
| $1$ (Fiedler only) | Maximum compression to single connectivity axis | High mode-collapse risk; strong basin selection; low `coverage` |
| $\approx F/4$ | Captures main basin boundaries, excludes noise | Regime where dual-space hypothesis predicts coverage gain |
| $F/2$ | Moderate spectral richness | Diminishing returns on basin structure |
| $F$ | $\Phi = \text{id}$, no projection | No spectral filtering benefit; correct non-spectral baseline for $\Phi$ specifically |

**Sweep protocol:** add $q \in \{1, 2, F/4, F/2, F\}$ as a column in
Experiment 9's diversity/quality table (`002-EXPERIMENTS.md`). The
primary diagnostic is `mode_separation` (std of per-sample Rayleigh
quotients $R(f_i)$ across the generated set, defined in
`03-FITNESS-CRITERIA.md` §Diversity). If mode separation peaks at
$q \approx F/4$ and collapses at both extremes, this is direct evidence
for the optimal lifting subspace size. The $q = F$ condition is also
the correct non-spectral baseline for $\Phi$ in isolation.

**Default for implementation:** $q = F/4$, rounded to nearest power of 2.
Override with `--q` flag in `verify_fitness.py`.

---

### Q3 — Asymmetry of $\Pi$ (contractive vs. reversible)

> **RESOLVED (v0.2.0, issue #23). $\Pi_\pm$ is not needed and should not
> be implemented.**

The concern was that the purely contractive $-\nabla R$ could not be inverted
for the reverse diffusion direction. This is now a non-issue: the reverse
path implemented in `esdm/sampling/entropic_sampler.py` uses the **DDPM
ancestral update rule** modulated by $\Lambda(\tau)$. The vibrational block's
`_hybridisation()` method is **not called during generation** — the model
runs `forward()` as a denoising predictor and the DDPM rule handles the
backward trajectory independently.

The `pi_contraction_ratio` diagnostic remains relevant for monitoring
forward-pass health:

$$\text{pi\_contraction\_ratio} = \frac{\|\Delta Q_t\|}{\|Q_t\|}$$

Healthy range $\in [0.01, 0.3]$ per step. Values $> 0.5$ indicate $\Pi$ is
overwhelming the wave dynamics (over-contraction); values near $0$ indicate
$\Pi$ is inactive (feature diffusion not reaching item space).

---

### Q4 — Interaction between $\Lambda(\tau)$ and $\mathcal{H}$

**Status: both diagnostics implemented; `net_energy_flow = 0` identified as
the current physics gap (fitness 0/5 at v0.2.0). Tracked in Issue #29.**

In ESDM the tension between the two opposing forces is:

$$\underbrace{+\Lambda(\tau)\cdot\Delta\tau^2\cdot B_t}_{\text{exploration: drives }Q_t\text{ away from smooth modes}}
\quad\text{vs}\quad
\underbrace{-\eta\,\Delta\tau^2\,\nabla_{Q_t} R(\tilde{F}_t)}_{\mathcal{H}\text{: pulls }Q_t\text{ toward smooth modes}}$$

`net_energy_flow` and `pi_contraction_ratio` are both implemented in
`VibrationalStateBlock.diagnostics()` and surfaced in `ESDM.health_check()`.
The current known weakness is that `net_energy_flow` is persistently near 0
rather than oscillating $\pm$ as theory predicts (Barontini Fig. 2 analogy).
This is the primary target for Issue #29.

$$\text{net\_energy\_flow}_t = \|Q_{t+1}\|_F^2 - \|Q_t\|_F^2$$

| Value | Interpretation |
|---|---|
| $\approx 0$ | Balanced: $\Lambda$ and $\mathcal{H}$ are in equilibrium |
| $> 0.1 \cdot \|Q_0\|_F^2$ | Forcing dominates: $\mathcal{H}$ is not pulling back fast enough |
| $< -0.5 \cdot \|Q_0\|_F^2$ | Contraction dominates: modes collapsing, generation diversity lost |

---

### Q5 — Relation to the signed density matrix ($\rho_-$ and negative Rayleigh curvature)

**Status: unvalidated analogy. Experiment 4 Condition C blocked on Issue #29.
`rectify_lifting` toggle is the pending implementation prerequisite.**

The geometric interpretation is that $\rho_- > \rho_+$ (destructive
interference dominates) means item states are in the *negative-curvature*
regions of the Rayleigh quotient surface — $\Pi$ is pointing the wrong
way, the restoring force is anti-aligned with the basin gradient.

Issue \#27 Gap 6 frames this more sharply: the question is whether
**anti-phase interactions** (negative Laplacian eigenvalue contributions,
equivalent to negative coupling sign in Un-0) produce *measurable* gains
in mode separation or controllability, or whether they are an interpretation
layer over a conventional result.

**The critical test is Experiment 4 Condition C** (`002-EXPERIMENTS.md`):
the unsigned spectral gate ablation replaces $F_t$ (which can have negative
components from anti-phase Laplacian modes) with $|F_t|$ (rectified,
no destructive interference). The `signed_vs_unsigned_delta` metric
(`03-FITNESS-CRITERIA.md` §B) measures the Chamfer distance delta between
the two conditions.

**Implementation prerequisite (not yet done):** add
`rectify_lifting: bool = False` to `VibrationalStateBlock.__init__()`.
When `True`, replace `F_t = Q_2d @ P` with `F_t = (Q_2d @ P).abs()` in
`_hybridisation()`. This is the Condition C toggle. No other code changes
are needed. This should be implemented when Issue #29 closes and
generation metrics are available to measure `signed_vs_unsigned_delta`.

**Decision rule:**
- `signed_vs_unsigned_delta` $\approx 0$ and `mode_separation` A ≈ C:
  $\rho_-$ is architecturally present but computationally inert. The
  quantum-interference framing should be replaced with a simpler
  "Laplacian diffusion with learned gate" description.
- `signed_vs_unsigned_delta` $> 0$ (signed better) and
  `mode_separation` A $>$ C: anti-phase dynamics are contributing real
  discriminative capacity. The signed density matrix interpretation is
  validated and `rho_sign` should be promoted to a first-class training
  signal (not just a confidence flag).
- `signed_vs_unsigned_delta` $< 0$ (unsigned better): apply a
  rectification layer $|\cdot|$ after $\Phi$ as a permanent
  architectural fix.

**Until Experiment 4 Condition C is run, the signed density matrix
interpretation is an analogy, not a validated mechanism.**

---

## 8. Relation to `00-DESIGN.md`

This document adds a subsection that should be read as **§4.5** of
`00-DESIGN.md`, inserted after the existing §4.4 (Entropic Clock as
Principled Collocation Strategy):

> **§4.5 The Hybridisation Operator $\mathcal{H}$**
> See `00-DESIGN_HYBRID.md` for the full derivation. In summary:
> the term $L_F Q_t$ in the wave recurrence is not a bare
> matrix–vector product. It is the composition
> $\mathcal{H}(Q_t) = \Pi \circ \text{Diffuse}_{L_F} \circ \Phi$
> where $\Phi$ lifts $Q_t$ into the feature-space eigenbasis,
> $\text{Diffuse}_{L_F}$ applies the Laplacian heat step,
> and $\Pi$ converts the result into a Rayleigh-gradient restoring force
> on the item-space state. The three operators together define the
> mechanism by which the F×F feature-space spectral structure (dark sector)
> directs the N×F geometric item-space dynamics (bright sector).
> Open questions and their v0.2.0 status are in `00-DESIGN_HYBRID.md` §7.

---

## References

- `00-DESIGN.md` §4.2–4.4 — wave recurrence, collocation
- `001-IMPLEMENTATION-PLAN.md` §4.2–4.3 — 2.5-D dual-space update equations
- `00-DESIGN_PREAMBLE.md` — two-sector partition
- `03-FITNESS-CRITERIA.md` §A–B — gate learning and interference scoring
- `002-EXPERIMENTS.md` Experiment 4 Condition C, Experiments 7–9
- Issue \#20: Hybridisation operator implementation (complete)
- Issue \#23: Generation pipeline / reverse-SDE sampler (complete)
- Issue \#27: Benchmark ESDM against Un-0-style oscillator dynamics
- Issue \#29: Generation validation (open — blocks Experiment 4 Condition C)
- ArrowSpace taumode: $R(f) = f^\top L_F f / f^\top f$, via `WireGraph.rayleigh_quotient`
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*.
  tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
