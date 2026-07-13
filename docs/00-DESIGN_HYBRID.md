# 00-DESIGN_HYBRID — The Hybridisation Operator

> **Status:** Design specification  
> **Depends on:** `00-DESIGN.md` §4, `001-IMPLEMENTATION-PLAN.md` §4.2–4.3  
> **Addresses:** the implicit mixing step between the F×F feature-space
> Laplacian signal and the N×F geometric item-space state that was
> embedded silently inside the wave recurrence in `00-DESIGN.md` §4.2/4.3
> but never written as a named, closed-form operator.

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

$$\boxed{\tilde{F}_t = F_t - \frac{\sigma^2 \Delta\tau}{2}\,\widetilde{L_F F_t}}$$

where $\widetilde{L_F F_t} = H_F^\top\!\left(g \odot
\left(\hat{L}_F \hat{F}_t^\top\right)^\top\right)$ is the WHT-filtered
Laplacian action (from `00-DESIGN.md` §4.3), with
$\hat{L}_F = H_F L_F H_F^\top$ precomputed and cached,
$\hat{F}_t = H_F F_t$, and $g \in \mathbb{R}^F$ the learnable spectral gate.

**Interpretation.** Each item's lifted feature signal diffuses along the
F×F feature-space graph. Items whose feature signals are far from the
graph's smooth modes are pulled toward them; items already in a smooth
mode are held there. The WHT gate $g$ controls which sequency bands of
$L_F$ are allowed to drive this diffusion — low sequency (globally coherent
string modes) are preserved; high sequency (locally noisy modes) are
progressively suppressed as $\tau$ increases.

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
row-wise across the batch, and $\eta > 0$ a learnable or fixed step size.

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
  - \tfrac{\sigma^2\Delta\tau}{2}\,\widetilde{L_F (Q_t U_q U_q^\top)}
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
              │  Diffuse: WHT-filtered L_F heat step ◄─┘
              ▼
        F̃_t  ∈  R^{B×F}
              │
              │  Π: Rayleigh gradient → restoring force
              ▼
        ΔQ_t  ∈  R^{B×F}
              │
              ▼
        Q_{t+1} = wave recurrence(Q_t, ΔQ_t, Λ(τ)·B_t)
```

---

## 5. Parameter Summary

| Symbol | Shape | Source | Learned? |
|---|---|---|---|
| $U_q$ | $F \times q$ | `SpectralLoadingDecoder` eigenvectors | No (fixed after graph build) |
| $q$ | scalar | hyperparameter ($1 \leq q \leq F$) | No |
| $\hat{L}_F = H_F L_F H_F^\top$ | $F \times F$ | precomputed in `build_laplacian.py` | No |
| $g$ | $F$ | learnable gate in `VibrationalStateBlock` | **Yes** |
| $\sigma^2$ | scalar | diffusion coefficient (fixed or annealed) | Optional |
| $\eta$ | scalar | Rayleigh gradient step size | Optional |
| $\Lambda(\tau)$ | scalar | `EntropicClock` output | No (derived) |

---

## 6. Degenerate and Special Cases

| Condition | Effect |
|---|---|
| $q = F$, $g = \mathbf{1}$ | $\Phi = \text{id}$, no WHT filtering → recovers the implicit `00-DESIGN.md` §4.2 formulation |
| $\sigma^2 \to 0$ | Diffuse step vanishes → $\tilde{F}_t = F_t$ → hybridisation reduces to pure Rayleigh-gradient restoring force on the projected state |
| $\eta \to 0$ | Projection step vanishes → feature-space diffusion has no effect on item dynamics → dark and bright sectors decouple |
| $q = 1$ | Only the Fiedler vector is used → maximum compression; item dynamics are governed entirely by the graph's primary connectivity axis |
| $g_k = 0$ for $k > k^*$ | Hard low-pass filter: only the $k^*$ lowest-sequency Laplacian modes are allowed to direct diffusion → deterministic pass-band |

---

## 7. Open Questions — Resolved Against Issue \#27 Findings

### Q1 — Backpropagation through $\Phi$ (differentiable $U_q$)

**Resolution: keep $U_q$ fixed through Experiments 7–9; differentiable $U_q$
is a Phase 3 variant.**

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
  itself vs. just the sequency gate $g$.

**Implementation note:** if $U_q$ is eventually made differentiable, the
correct approach is a Lanczos-based differentiable eigendecomposition
(e.g. `torch.lobpcg` with `largest=False`) rather than full
`torch.linalg.eigh`, since only the leading $q \ll F$ eigenvectors are
needed. Mark as `# TEMPORAL: upstream issue` per `001-IMPLEMENTATION-PLAN.md`
§3 rules until a native `arrowspace` differentiable eigenbasis lands.

---

### Q2 — Choice of $q$

**Resolution: $q$ is not a free scalar hyperparameter — it must be swept
jointly with the diversity/quality split (Experiment 9).**

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

**Default for implementation:** $q = F/4$, rounded to nearest power of 2
for WHT compatibility. Override with `--q` flag in experiment configs.

---

### Q3 — Asymmetry of $\Pi$ (contractive vs. reversible)

**Resolution: the asymmetry is a feature for forward diffusion but must
be explicitly parameterised for the Loschmidt echo test.**

The Rayleigh gradient $-\nabla_{Q_t} R$ is purely contractive: it always
pulls item states toward lower spectral energy. This is correct for the
*forward* diffusion direction (smoothing, semantic basin descent) but
incompatible with the *reverse* direction (reconstruction, generation
from noise) if the reverse process must climb the Rayleigh surface.

Issue \#27 Gap 3 (endpoint-vs-trajectory supervision) is directly
relevant: if supervising only terminal states is sufficient, the
asymmetry is irrelevant during training — the reverse process never needs
to invert $\Pi$. If intermediate states matter, the contraction must be
reversible.

**Resolution for the Loschmidt echo** (`00-DESIGN.md` §11.4): define a
**signed projection** $\Pi_\pm$ controlled by the sign of $\Delta\tau$:

$$\Pi_\pm(\tilde{F}_t,\, \text{sign}(\Delta\tau))
  = -\,\text{sign}(\Delta\tau)\;\eta\,\nabla_{Q_t} R(\tilde{F}_t)$$

This uses $-\nabla R$ in the forward direction (contraction, basin descent)
and $+\nabla R$ in the reverse direction (expansion, basin ascent).
Structurally this mirrors how score-based reverse SDEs flip the drift sign.
The asymmetry is not resolved — it is made explicit and parameterised.

**Monitoring:** add `pi_contraction_ratio` to the health check:

$$\text{pi\_contraction\_ratio} = \frac{\|\Delta Q_t\|}{\|Q_t\|}$$

Healthy range $\in [0.01, 0.3]$ per step. Values $> 0.5$ indicate $\Pi$ is
overwhelming the wave dynamics (over-contraction); values near $0$ indicate
$\Pi$ is inactive (feature diffusion not reaching item space).

---

### Q4 — Interaction between $\Lambda(\tau)$ and $\mathcal{H}$

**Resolution: the $\Lambda$ vs $\mathcal{H}$ tension is the ESDM analogue
of the Un-0 coupling-strength ablation; add `net_energy_flow` to the
health check table.**

In ESDM the tension between the two opposing forces is:

$$\underbrace{+\Lambda(\tau)\cdot\Delta\tau^2\cdot B_t}_{\text{exploration: drives }Q_t\text{ away from smooth modes}}
\quad\text{vs}\quad
\underbrace{-\eta\,\Delta\tau^2\,\nabla_{Q_t} R(\tilde{F}_t)}_{\mathcal{H}\text{: pulls }Q_t\text{ toward smooth modes}}$$

This is structurally identical to the Un-0 coupling-strength parameter: in
Un-0 the coupling matrix magnitude controls how hard oscillators pull each
other toward synchrony vs. how hard the external input drives them apart.
Issue \#27 Gap 1 asks for the frozen-vs-learned ablation precisely to
isolate this balance.

**`r_g_corr`** (residual-gate correlation, `03-FITNESS-CRITERIA.md` §B)
measures the per-mode correlation between residual magnitude and gate
activity — it is a proxy for whether $\mathcal{H}$ is responding to
meaningful spectral activity. But it does not directly measure which force
is winning globally.

**Add `net_energy_flow` to the health check** (`00-DESIGN.md` §8.1):

$$\text{net\_energy\_flow}_t = \|Q_{t+1}\|_F^2 - \|Q_t\|_F^2$$

| Value | Interpretation |
|---|---|
| $\approx 0$ | Balanced: $\Lambda$ and $\mathcal{H}$ are in equilibrium |
| $> 0.1 \cdot \|Q_0\|_F^2$ | Forcing dominates: $\mathcal{H}$ is not pulling back fast enough |
| $< -0.5 \cdot \|Q_0\|_F^2$ | Contraction dominates: modes collapsing, generation diversity lost |

The `kl_S` barrier-height sweep (`00-DESIGN.md` §9) will produce one
`net_energy_flow` curve per barrier height, directly analogous to
Barontini's Fig. 2 (entropic time vs. barrier height). Low `kl_S` should
produce oscillating $\pm$ energy flow (cyclic expansion/contraction); high
`kl_S` should produce monotonically negative flow (rapid heat death).

---

### Q5 — Relation to the signed density matrix ($\rho_-$ and negative Rayleigh curvature)

**Resolution: $\rho_-$ is currently an interpretive analogy; it becomes a
validated mechanism only if Experiment 4 Condition C (unsigned spectral gate)
shows a measurable delta. Until then, the signed density matrix is flagged
as unvalidated.**

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

**Decision rule:**
- `signed_vs_unsigned_delta` $\approx 0$ and `mode_separation` A ≈ C:
  $\rho_-$ is architecturally present but computationally inert. The
  quantum-interference framing should be replaced with a simpler
  "Laplacian diffusion with learned gate" description. No action needed
  on the signed density matrix.
- `signed_vs_unsigned_delta` $> 0$ (signed better) and
  `mode_separation` A $>$ C: anti-phase dynamics are contributing real
  discriminative capacity. The signed density matrix interpretation is
  validated and `rho_sign` should be promoted to a first-class training
  signal (not just a confidence flag).
- `signed_vs_unsigned_delta` $< 0$ (unsigned better): the signed
  coefficients are harmful — anti-phase interference is corrupting the
  diffusion trajectory. In this case, apply a rectification layer
  $|\cdot|$ after $\Phi$ as a permanent architectural fix.

**Until Experiment 4 Condition C is run, the signed density matrix
interpretation is an analogy, not a validated mechanism.** This is the
uncomfortable but correct reading of the issue \#27 findings.

---

## 8. Relation to `00-DESIGN.md`

This document adds a subsection that should be read as **§4.5** of
`00-DESIGN.md`, inserted after the existing §4.4 (Entropic Clock as
Principled Collocation Strategy):

> **§4.5 The Hybridisation Operator $\mathcal{H}$**
> See `00-DESIGN_HYBRID.md` for the full derivation. In summary:
> the term $\widetilde{L_F Q_t}$ in the wave recurrence is not a bare
> matrix–vector product. It is the composition
> $\mathcal{H}(Q_t) = \Pi \circ \text{Diffuse}_{L_F} \circ \Phi$
> where $\Phi$ lifts $Q_t$ into the feature-space eigenbasis,
> $\text{Diffuse}_{L_F}$ applies the WHT-filtered Laplacian heat step,
> and $\Pi$ converts the result into a Rayleigh-gradient restoring force
> on the item-space state. The three operators together define the
> mechanism by which the F×F feature-space spectral structure (dark sector)
> directs the N×F geometric item-space dynamics (bright sector).
> Open questions resolved against issue \#27 findings are in
> `00-DESIGN_HYBRID.md` §7.

---

## References

- `00-DESIGN.md` §4.2–4.4 — wave recurrence, WHT filter, collocation
- `001-IMPLEMENTATION-PLAN.md` §4.2–4.3 — 2.5-D dual-space update equations
- `00-DESIGN_PREAMBLE.md` — WHT spectral filter, two-sector partition
- `03-FITNESS-CRITERIA.md` §A–B — gate learning and interference scoring
- `002-EXPERIMENTS.md` Experiment 4 Condition C, Experiments 7–9
- Issue \#27: Benchmark ESDM against Un-0-style oscillator dynamics
- ArrowSpace taumode: $R(f) = f^\top L_F f / f^\top f$, via `WireGraph.rayleigh_quotient`
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*.
  tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
