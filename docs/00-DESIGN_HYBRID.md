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

## 7. Open Questions

1. **Backpropagation through $\Phi$.** $U_q$ is currently fixed after graph
   construction. If $U_q$ is made differentiable (e.g. via a differentiable
   eigendecomposition or a Lanczos approximation), the lifting step can be
   jointly optimised with the rest of the model. This is flagged as an open
   risk in `001-IMPLEMENTATION-PLAN.md` §7.

2. **Choice of $q$.** In the 2.5-D track, $q = 2$ (Fiedler + second
   eigenvector). For the full ESDM, $q$ should be treated as a
   hyperparameter to sweep: too small removes information, too large
   (approaching $F$) removes the spectral projection benefit.

3. **Asymmetry of $\Pi$.** The Rayleigh gradient $\nabla_{Q_t} R$ points
   toward lower spectral energy. This means the hybridisation operator
   always pulls item states toward smoother feature-space modes — it is a
   **contractive** operator, not a conservative one. Whether this is
   desirable or whether a symmetric (reversible) projection is needed for
   the Loschmidt echo reversibility test (`00-DESIGN.md` §11.4) is an
   open question.

4. **Interaction with $\Lambda(\tau)$.** The forcing term $\Lambda(\tau) B_t$
   can drive item states away from smooth modes (when $\Lambda$ is large).
   The hybridisation operator $\mathcal{H}$ then pulls them back. The
   tension between $\Lambda$ (exploration, attention-driven) and
   $\mathcal{H}$ (contraction, Rayleigh-driven) is the core dynamical
   balance of ESDM and should be monitored via the `r_g_corr` diagnostic
   (`03-FITNESS-CRITERIA.md` §B).

5. **Relation to the signed density matrix.** Destructive interference
   ($\rho_- > \rho_+$) occurs when $\mathcal{H}$ is driving item states
   into anti-phase configurations relative to the Laplacian eigenmodes.
   The `rho_sign` confidence signal (`00-DESIGN.md` §6.3) can therefore
   be interpreted geometrically: high $\rho_-$ means items are
   systematically positioned in the negative-curvature regions of the
   Rayleigh quotient surface — the restoring force is pointing the
   wrong way. See `03-FITNESS-CRITERIA.md` §B for the operationalisation.

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

---

## References

- `00-DESIGN.md` §4.2–4.4 — wave recurrence, WHT filter, collocation
- `001-IMPLEMENTATION-PLAN.md` §4.2–4.3 — 2.5-D dual-space update equations
- `00-DESIGN_PREAMBLE.md` — WHT spectral filter, two-sector partition
- `03-FITNESS-CRITERIA.md` §A–B — gate learning and interference scoring
- ArrowSpace taumode: $R(f) = f^\top L_F f / f^\top f$, via `WireGraph.rayleigh_quotient`
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*.
  tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
