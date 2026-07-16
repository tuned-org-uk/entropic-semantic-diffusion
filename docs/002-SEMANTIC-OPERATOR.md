# 002 — The Semantic Operator

> **Status:** v0.4.0 — topology-adaptive decoder (issue #24)
> **Depends on:** `docs/00-DESIGN.md` (§1, §4, §12), `docs/001-IMPLEMENTATION-PLAN.md` (§5, §6.3, §9)
> **Audience:** a reader who knows diffusion models but not this codebase. This is the entry point for understanding *what ESDM is, what it currently does, and what it must do to become a generative model.*

---

## 1. The concept in one paragraph

ESDM is a **vibrational** diffusion model. It is not a score-matching model with an entropic clock bolted on; the clock, the operator, and the geometry are one mechanism. The model is built from three ingredients:

1. **A geometry space.** An $L_2$-normalised embedding corpus $V \in \mathbb{R}^{N \times F}$ defines a feature-space graph Laplacian $L_F \in \mathbb{R}^{F \times F}$ — the spectral geometry of the semantic field. Its leading eigenpairs $(U_q, \Lambda_q)$ are the smooth modes of that field. This is the *manifold on which diffusion happens*. It is built once, frozen, and never learned.
2. **ArrowSpace** — the structure carried by that graph: the Laplacian itself (`gl`, the structural connectivity) **and** the per-item energy distribution (`lambdas()`, where each item sits on the energy landscape: low lambda = smooth / on-manifold, high lambda = rough / off-manifold). ArrowSpace is *both* the matrix and the distribution; the lambdas are not a post-hoc weighting, they are half the space.
3. **Vibrational deductive dynamics.** The item-space state $Q_t \in \mathbb{R}^{B \times F}$ evolves on $S^{F-1}$ via a Rayleigh-string wave recurrence whose restoring force is the gradient of the Dirichlet energy on the graph, and whose time is *internal* — the entropic clock $\tau_n = \sum |\Delta\,\mathit{kl\_tau}|$, not a step counter.

The **semantic operator** $\mathcal{H}$ is the general concept that fuses these three; the **encoder–decoder** is the diffusion implementation of that concept, applied on both sides of the generative process. Both are vibrational: both act on the same ArrowSpace geometry, both use the same spectral chart, both invert through the same eigenbasis. This is what makes ESDM a *vibrational* diffusion model rather than a score-matching diffusion model with a graph prior.

$$
\text{ESDM} \;=\; \underbrace{\text{geometry space}}_{L_F,\;U_q} \;+\; \underbrace{\text{ArrowSpace}}_{\textit{gl}\;+\;\textit{lambdas}} \;+\; \underbrace{\text{vibrational deductive}}_{\text{wave recurrence on } S^{F-1}}
$$

### 1.1 Literature

- **Rayleigh, *Theory of Sound* (1877).** The restoring force of a stretched string is the gradient of the Rayleigh quotient $R(f) = f^\top L f / f^\top f$. The $\Pi$ step of $\mathcal{H}$ is exactly this gradient, computed on the feature-space graph instead of a continuous string.
- **Barontini (2026), *Testing the problem of time with cold atoms*, Phys. Rev. Research 8, L022047.** A closed system with a time-independent Hamiltonian has no built-in clock; an *entropic time* derived from internal entropy exchange between a bright and a dark sector replaces the external laboratory clock. The ESDM's $\tau_n$ is this entropic time, with the spectral KLs playing the role of sector entropies.
- **Moriondo (2024), *ArrowSpace: Semantic Basins* / *Graph Wiring*.** Every vector space is simultaneously a geometric item-space and a semantic feature-space. The feature-space Laplacian and its lambda distribution are the dual-space structure; $U_q$ is the spectral embedding of that duality.
- **VDT (vibrational diffusion transport), Zenodo 20816835** and **arXiv 2606.21535.** The wave recurrence, the spectral loading decoder, the signed density matrix, and the CFL machinery are the VDT primitives the ESDM inherits. The ESDM contribution is the entropic clock, the hybridisation operator $\mathcal{H}$, and the topology-adaptive reverse decoder.

---

## 2. How ESDM differs from other diffusion models

This is the section a third party should read first.

| | DDPM / EDM (score-matching) | ESDM (vibrational) |
|---|---|---|
| **Time** | External step index $t \in [0, T]$ | Internal entropic time $\tau_n = \sum\|\Delta\mathit{kl\_tau}\|$, monotone by construction, stops at heat death |
| **Forward (noise) operator** | Isotropic Gaussian $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ | Laplacian heat flow on the ArrowSpace graph: $\tilde F_t = F_t - \tfrac{\sigma^2 \Delta\tau}{2} L_F F_t$ (parabolic, anisotropic, geometry-directed) |
| **Score** | Learned $\nabla \log p_t(x)$ via denoising score matching | Rayleigh gradient $-\nabla R$ on the graph — *variationally equivalent* to the score toward smoother (lower-Dirichlet-energy) configurations on the wired manifold (§12.7 of the design) |
| **Restoring force** | Implicit in the learned score network | Explicit: $\mathcal{H}(Q_t) = \Pi(\text{Diffuse}_{L_F}(\Phi(Q_t)))$ inside the wave recurrence |
| **Generation space** | Ambient $\mathbb{R}^F$; off-manifold samples possible | Spectral chart $\hat q \in \mathbb{R}^K$ (the $U_q$ coordinates); lifted back via $\Phi_K \hat q$; on-manifold *by construction* once the reverse decoder is wired |
| **Schedule** | Fixed cosine / sigmoid in $t$ | $\bar\alpha(\tau) = \cos^2(\tfrac{\pi}{2} \tau/\tau_\mathit{max})$, gated by accumulated entropy; *the clock decides where to evaluate the physics* |
| **Determinant / Jacobian** | Required for continuous Fokker–Planck volume tracking | Dropped — Dirichlet energy is a quadratic form on the graph, not a local volume distortion (§12.7) |

The single sentence: **in a score-matching diffusion model the network learns the score of a Gaussian-noised data distribution; in ESDM the network *executes* a wave equation whose restoring force is the gradient of the data's own graph energy, and whose time is the entropy that flow produces.** The learned parameters ($\sigma^2, \eta, B_t$) shape the dynamics on top of the frozen geometry; they do not replace it.

---

## 3. The semantic operator — the general concept

The semantic operator $\mathcal{H}$ is the abstract mechanism that fuses the geometry, ArrowSpace, and the vibrational dynamics. It is *one* object, defined as a three-step procedure on the ArrowSpace geometry:

$$
\mathcal{H}(Q_t) \;=\; \Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)
$$

1. **Lifting $\Phi$** — $F_t = Q_t\, U_q U_q^\top$. Project the item state onto the smooth eigenbasis of $L_F$ and lift back. Components orthogonal to $U_q$ (high-frequency, locally incoherent) are suppressed before the Laplacian acts. $\Phi$ restricts the system to the semantically coherent subspace of feature space.
2. **Diffusion $\text{Diffuse}_{L_F}$** — $\tilde F_t = F_t - \tfrac{\sigma^2 \Delta\tau}{2}\, L_F F_t$. A parabolic heat step along the feature-space graph. Items far from smooth modes are pulled toward them; items already smooth are held. The $\sigma^2 \Delta\tau / 2$ (heat-equation) coefficient, not $\Delta\tau^2$ (wave-equation), is deliberate: this step is *parabolic* (irreversible smoothing); the wave recurrence that consumes the output is *hyperbolic* (oscillatory). The operator combines both characters.
3. **Projection $\Pi$** — $\Delta Q_t = -\eta\, \nabla_{Q_t} R(\tilde F_t)$, the Rayleigh-gradient restoring force. Items with high Rayleigh quotient (high spectral energy) feel a large force toward low-energy regions; items already smooth feel ~zero. This is the string/spring restoring force from Rayleigh's *Theory of Sound*, computed on the graph instead of a continuous string.

The operator is **direction-agnostic**: it is the procedure (project, diffuse along the graph, convert to a Rayleigh-gradient force), not a choice of encoding or decoding. The encoding and the decoding are *implementations* of this concept on opposite sides of the generative process — see §4. The frozen geometric prior ($L_F$, $U_q$, the lambdas) defines the manifold; the learned structure ($\sigma^2$, $\eta$, $B_t$) shapes the dynamics on it. The operator is what makes that distinction operational rather than decorative.

### 3.1 The $\Lambda/\mathcal{H}$ balance

The operator does not act alone. The wave recurrence that consumes its output is

$$
Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2\,\mathcal{H}(Q_t) - \gamma\,\Delta Q + \Lambda(\tau)\cdot\Delta\tau^2\,B_t
$$

where $\Lambda(\tau) = |\partial_\phi\,\mathit{kl\_tau}|^{-1}$ is the **entropic pump** and $B_t$ is the attention-forcing residual. The tension between the pump $\Lambda$ (exploration — drives $Q_t$ away from smooth modes) and $\mathcal{H}$ (restoration — pulls $Q_t$ toward smooth modes) *is* the dynamics. When they are balanced, the system oscillates; when the pump dominates, the $Q$-state is destroyed; when $\mathcal{H}$ dominates, the system collapses to a single mode. The design constraint is that the pump force and the $\mathcal{H}$ restoring force must be the same order of magnitude throughout training. This balance is the open forward-path blocker — see §6.

---

## 4. The diffusion implementation — encoder and decoder

The diffusion model implements the semantic operator *twice*: once as an **encoder** (the forward direction, training time) and once as a **decoder** (the reverse direction, generation time). Both instantiations operate on the same ArrowSpace geometry and the same spectral chart; they differ in what they do with the operator's output.

### 4.1 The encoder — vibrational deductive (training)

The encoding side is the **vibrational block**: the wave recurrence on $S^{F-1}$ with $\mathcal{H}$ as its restoring force. It is *vibrational deductive*: the next state is deduced from the current one by applying the operator, not predicted by a learned residual. This is the VDT applied to ArrowSpace — the wave recurrence and CFL machinery are VDT primitives; $L_F$ and $U_q$ are ArrowSpace primitives; $\mathcal{H}$ fuses them.

```
Item space (bright sector)               Feature space (dark sector)
Q_t  ∈  R^{B×F}  on S^{F-1}             L_F  ∈  R^{F×F}   (frozen)
      │                                         │
      │  Φ: project onto U_q subspace            │
      ▼                                         │
F_t = Q_t U_q U_q^T  ∈  R^{B×F}                │
      │                                         │
      │  Diffuse: L_F heat step ◄───────────────┘
      ▼
F̃_t  ∈  R^{B×F}
      │
      │  Π: Rayleigh gradient → restoring force
      ▼
ΔQ_t  ∈  R^{B×F}
      │
      ▼
Q_{t+1} = wave recurrence(Q_t, ΔQ_t, Λ(τ)·B_t)   then project back to S^{F-1}
```

The encoder runs during training. Its job is to shape the item-space trajectories toward semantic basins, gated by the entropic clock. It is the forward face of $\mathcal{H}$, and it is **implemented and wired** (issue #20, complete — `esdm/models/vibrational.py:255`).

### 4.2 The decoder — topology-adaptive (generation)

The decoding side is the **topology-adaptive decoder**: the same operator run backwards along the same geometry. Instead of producing a restoring force on $Q_t$, it produces an embedding on the smooth submanifold. It operates in the **spectral chart** $\hat q \in \mathbb{R}^K$ — the $K$-dimensional coordinates on $U_q$ — not in ambient $\mathbb{R}^F$.

```
Q ∈ R^F
  │  project onto Φ_K (U_q, the smooth eigenvectors)
  ▼
q̂ ∈ R^K          ← spectral chart (intrinsic manifold coordinates)
  │  heat_kernel_sample(L_K, t=ᾱ(τ))   with   L_K = U_q^T L_F U_q
  ▼
q̂_noisy ∈ R^K   ← manifold-aware noised sample (heat kernel prior)
  │  TopologyDecoder MLP
  ▼
x̂ ∈ R^F          ← reconstructed embedding, on the smooth submanifold by construction
```

The heat kernel prior $\hat q \sim \mathcal{N}(0, e^{-2t L_K})$ is the natural prior on the spectral chart. At $t \to 0$ samples concentrate near data; at $t \to \infty$ they spread over the full smooth submanifold. The $\bar\alpha(\tau)$ schedule *is* this $t$ parameter — the two are directly coupled. The lambda distribution biases the prior toward the smooth regime where real data lives (low lambda = high weight). Sampling in spectral coordinates is sampling on the manifold by construction: a point drawn from $\hat q \sim p(\hat q)$ and lifted back via $\Phi_K \hat q$ always lies on the smooth submanifold. Off-manifold generation becomes geometrically impossible — *once the decoder is wired*.

This is the implementation of the semantic operator in a diffusion model: the decoder *is* $\mathcal{H}$ run in reverse on the ArrowSpace geometry. It is **implemented and unit-tested** (`esdm/models/topology_decoder.py`, `tests/test_topology_decoder.py`) but **not yet wired into the training graph or the 2.5-D track** — this is the central gap the roadmap closes.

### 4.3 The 2.5-D dual-space track

The decoding side must work on the 2.5-D track. The encoding side already does: the vibrational block operates on a single-channel node-feature state $Q: (B, F, 1)$, which is the dual-space split — $F$ nodes (the feature-space graph) carrying one scalar each (the item coordinate). The 2.5-D treatment condition under test is whether a 2.5-D dual-space model (2-D item coordinate + graph-Laplacian feature signal) can match full-dimensional volumetric diffusion without computing Jacobian determinants. The `esdm/models/score25d.py` scaffold is the treatment condition; the topology decoder is its natural generation side. Connecting them tests the original research question — and the encoding side's geometry is already the 2.5-D geometry.

### 4.4 Why "vibrational" on both sides

Encoding: the wave recurrence deduces the next state by applying the restoring force — a deductive time step, not a learned residual. Decoding: the reverse step projects the generated state onto the spectral chart and lifts back — the *same* eigenbasis, the *same* graph, the inverse of the same heat flow. Neither side requires a learned score in ambient space; both sides use the ArrowSpace geometry as the substrate. The encoding side is wired (vibrational block). The decoding side is implemented as `TopologyDecoder` but not yet wired — see §5.4.

---

## 5. v0.4.0 implementation snapshot

What the code actually computes today. Every entry is grounded in a file reference.

### 5.1 Geometry — ArrowSpace (`esdm/graphs/wiring.py`)

`WireGraph.build(features, k, seed, mode="feature")` calls the ArrowSpace Rust bindings (`arrowspace==0.26.2`) and returns *both* halves of ArrowSpace:

- `laplacian_dense` $(F, F)$ — the `gl` structural connectivity.
- `lambdas` $(N,)$ — the per-item energy scores from `aspace.lambdas()`. Low = smooth / on-manifold, high = rough / off-manifold.

When the bindings are absent, a k-NN fallback is used (marked `# TEMPORARY: upstream issue`). The symmetric normalised Laplacian is cached. `rayleigh_quotient(f)` computes $R_k(f) = f_k (Lf)_k / (f^\top f)$ — the per-node Dirichlet-energy density, the $\lambda\tau$ proxy. `spectral_embedding(n_components)` returns the leading non-trivial eigenvectors (skipping the zero eigenvalue).

### 5.2 Encoder — the forward operator (`esdm/models/vibrational.py`)

`VibrationalStateBlock._hybridisation` (`vibrational.py:255`) implements all three steps of $\mathcal{H}$:

- $\Phi$: `P = U_q @ U_q.t(); F_t = Q_2d @ P` (line 291–292).
- Diffuse: `F_tilde = F_t - (sigma_sq * dt / 2) * (F_t @ L)` (line 296). Parabolic heat step with $\sigma^2 \Delta\tau / 2$, deliberately not the wave $\Delta\tau^2$ — Step 2 is irreversible smoothing, Step 3 is oscillatory.
- $\Pi$: row-wise Rayleigh quotient `R = (F_tilde * L_F_F_tilde).sum(-1) / norms_sq` (line 303), gradient `grad_R = (2/norms_sq) * (L_F_F_tilde - R * F_tilde)` (line 306), projected through $P$ and scaled by $\eta$.

**Sign convention** (`vibrational.py:270`): `_hybridisation` returns `H_action = eta * grad_R` (analogous to $L Q$). The recurrence applies $-\Delta\tau^2 \cdot \text{H\_action}$, giving the restoring force $-\eta \Delta\tau^2 \nabla R$ toward low-energy modes. The design-doc sign $\Delta Q_t = -\eta \nabla R$ and the code agree once the $-\Delta\tau^2$ factor in `_step()` is read with it.

**Geometric constraint** (`vibrational.py:385`): after each step `Q_tp1 = F.normalize(Q_tp1.squeeze(-1), dim=-1)` — projection back onto $S^{F-1}$. Without this the Rayleigh gradient shrinks (it is normalised by $\|\tilde F_t\|^2$) and the recurrence amplifies energy instead of bounding it.

**State**: `sigma_sq` and `eta` are `nn.Parameter` (learnable), `U_q` is a frozen buffer. `eta` is *derived at init* from $L_F$'s spectral scale: $\eta_0 = 0.1 F / (2\lambda_\mathit{max})$ (`esdm.py:129`), so the $\mathcal{H}$ operator's initial strength is dataset-agnostic. `n_steps = 2` (CFL-safe for real $L_F$).

### 5.3 The entropic clock (`esdm/models/clock.py`)

`EntropicClock.step` accumulates $\tau_n = \sum |\Delta\,\mathit{kl\_tau}|$ (raw, not normalised by $S_0$ — normalising froze the clock on CIFAR-10). It computes the pump $\Lambda(\tau) = 1/|\partial_\phi\,\mathit{kl\_tau}|$ clipped to $\lambda_\mathit{clip} = 0.1 F$ and gated by $\bar\alpha(\tau)$ (full strength early, ramping to zero late so $\mathcal{H}$ can pull $Q$ to smooth modes without pump interference). Heat death fires when $\tau_n$ stalls for `heat_death_patience` steps (dual absolute + relative threshold). `tau_max_est = 3 S_0` is a derived horizon, not a hyperparameter.

### 5.4 Decoder — the reverse operator (`esdm/models/topology_decoder.py`)

- `TopologyDecoder` (line 29): an MLP from $\mathbb{R}^K \to \mathbb{R}^F$ that operates on spectral coordinates. `encode_to_spectral(Q) = Q @ phi` and `lift_from_spectral(q_hat) = q_hat @ phi.T` are the chart projection and lift — the exact inverse of $\Phi$.
- `build_projected_laplacian(U_q, L_F) = U_q.T @ L_F @ U_q` (line 85) — the reduced Laplacian $L_K$ whose eigenvalues are the energy landscape of the spectral chart.
- `heat_kernel_sample(L_K, t, n_samples, seed, lambdas)` (line 99): draws from $\mathcal{N}(0, e^{-2t L_K})$ with optional lambda weighting (low lambda = high variance = bias toward smooth modes). Implemented and unit-tested (`tests/test_topology_decoder.py:209` confirms variance concentrates in low-lambda directions).
- `topology_reverse_sde(model, n_samples, n_steps, seed)` (line 226): the full reverse diffusion in spectral coordinates. Initialises from the heat kernel prior at $t = \tau_\mathit{max}$, steps $\tau_\mathit{max} \to 0$, lifts to ambient at the end. Uses `model._last_Q` from the forward pass as the score predictor (leapfrog approximation — the wave recurrence with nonlinear $B_t$ and $\mathcal{H}$ cannot be algebraically inverted).

### 5.5 The two samplers that predate the topology decoder

For completeness, since they exist in the repo:

- `esdm/sampling/entropic_sampler.py` — DDPM ancestral update in input space. Architecturally mismatched with ESDM (the diffusion is internal to $Q$, not to $x$); produces zero embeddings on real data. Kept for backward compatibility.
- `esdm/sampling/q_space_sampler.py` — score-approximation in $Q$-space with $U_q$-constrained initialisation. Produces non-zero embeddings with correct marginal stats but off-manifold (NN ratio ~74). The DDIM-style update uses the model forward pass as a score predictor.

The topology sampler is the architecturally correct one; the first two are stepping stones the design has superseded.

### 5.6 What is wired vs. what is not

| Component | Implemented | Wired into training | Wired into generation |
|---|---|---|---|
| Forward $\mathcal{H}$ (vibrational block, encoder) | yes | yes (`esdm.py:271`) | n/a (training only) |
| ArrowSpace $L_F$ + lambdas | yes | $L_F$ yes; lambdas stored on model, used in topology sampler | lambdas not yet integrated into the sampler loop |
| Entropic clock | yes | yes (`esdm.py:252`) | saved/restored by samplers |
| `TopologyDecoder` (reverse $\mathcal{H}$, decoder) | yes | **no** — training still uses the ambient `EntropicDiffusionDecoder` (`esdm.py:285`: $\hat x = \sqrt{\bar\alpha}\,Q$) | sampler-only (`topology_reverse_sde`); training signal never flows through it |
| 2.5-D dual-space score track (`score25d.py`) | scaffold | no | no |

This table is the single most important fact in the document. The reverse operator exists and is unit-tested, but the training loss does not yet flow through it. Until it does, the "complete forward–reverse pair" of §4.2 is a design claim, not a trained system.

---

## 6. The known forward-path blocker: $\Lambda/\mathcal{H}$ imbalance

Even on the encoding side, the dynamics are not yet balanced. From the v0.2.0 assessment (still current as of v0.4.0):

- `net_energy_flow = 0.00 / 5` — the $\Lambda$ pump dominates the $\mathcal{H}$ contraction throughout training. The design (§3.1) says the tension between pump (exploration) and $\mathcal{H}$ (restoration) *is* the dynamics; in v0.4.0 the tension is one-sided domination.
- `recon_decrease = 6.58 / 10` — the ambient decoder is near-collapsed (recon $0.011 \to 0.0006 \to 0.004$). The metric rewards a tiny absolute decrease in an already-small quantity.
- `pi_contraction_ratio = 3.94 / 5` — $\Pi$ activity is $>1.0$ at start (too strong) and $\to 0.02$ at end (inert). $\mathcal{H}$ is too strong early, inert late.

This is a design-level problem, not a tuning problem. The $\lambda_\mathit{clip} = 0.1 F$ derivation (§4.6 of the design) is the fix for the *initial* balance; the *running* balance — keeping the pump and $\mathcal{H}$ the same order of magnitude throughout training — is the open issue (#29 Phase 2). Until this is resolved, the forward operator cannot be trusted as the encoding side of a generative model, because the dynamics it encodes are not the design's dynamics.

---

## 7. Roadmap

Two horizons. The short-term closes the wiring gap and the balance bug. The long-term delivers a generative model a third party can evaluate.

### 7.1 Short-term (next ~2 versions) — wire the decoder, fix the balance

1. **Route the training loss through `TopologyDecoder`.** Replace the ambient `EntropicDiffusionDecoder` call at `esdm.py:285` with the topology decoder so the reconstruction loss is computed on the spectral chart, not on $\sqrt{\bar\alpha}\,Q$. The decoder already implements the chart projection, MLP, and lift. This is the single change that makes the encoder–decoder a *trained* pair rather than a designed pair. Risk: the MLP adds parameters the design's "constants over parameters" principle (AGENTS.md first principle 2) must scrutinise — the MLP width and depth need design justification, not defaults.
2. **Integrate the lambda distribution into the sampler loop.** `heat_kernel_sample` accepts `lambdas` and `empirical_spectral_prior` weights by $1 - \text{lambdas}$, but `topology_reverse_sde` currently derives `lambdas_K` from `eigvals_q` rather than from the model's stored ArrowSpace `lambdas` (`topology_decoder.py:279`). Wire the actual `model.lambdas` through so the prior reflects the data's energy distribution, not just the graph's structural connectivity. This is the difference between sampling on the graph and sampling on ArrowSpace.
3. **Resolve the $\Lambda/\mathcal{H}$ imbalance (#29 Phase 2).** The pump starts at the clip (25.6 for $F=256$) and dominates $\mathcal{H}$ at init. The $\bar\alpha$-gating already in the clock (`clock.py:120`) is the right structural fix; the remaining work is verifying that `net_energy_flow` goes negative (pump suppressed, $\mathcal{H}$ dominates) in the final phase and oscillates around zero. This is the diagnostic that replaces the tautological `conservation = 15/15` (enforced by construction — `dark_entropy = max(0, S_0 - kl_S - kl_tau)`).
4. **Validate on the 2.5-D track.** The decoding side must work on the dual-space track: the `score25d.py` scaffold (item coordinate $z_t$ + feature signal $f_t$ on the wired graph) is the treatment condition. The topology decoder is its natural generation side. Connecting them tests whether a 2.5-D model can match full-dimensional diffusion without Jacobian determinants — the original research question.

### 7.2 Long-term — fully-fledged generation

The delivery criterion is not a fitness score. It is a generation validation gate:

1. **On-manifold generation.** The nearest-neighbour ratio (generated-to-reference vs reference-to-reference distance) must be $\approx 1$. v0.4.0 topology sampler is at NN ratio ~81 (off-manifold). The gate is NN ratio $\in [0.9, 1.1]$ on CIFAR-10 DINO-256. This is the §4.2 claim ("off-manifold generation becomes geometrically impossible") made empirical.
2. **Diversity, not just precision.** The design (§11) defines `coverage` (fraction of reference samples with a generated neighbour within Chamfer threshold $\delta$) and `precision` (the reverse). An improvement in `precision` without `coverage` is *not* a win for the dual-space hypothesis — the hypothesis predicts improved basin coverage, i.e. higher `coverage` and higher `mode_separation` ($\text{std}(R(f_i))$ across generated samples), not merely sharper samples. The generative evaluation must report both.
3. **FID / visual inspection.** Decode generated embeddings to pixels and compute FID against the reference set. The pixel decoder and CLI exist (#23, complete) but are untested on real data (#29 Phase 1).
4. **Ablation against a non-spectral baseline (#27).** Replace $L_F$ with $I$ (no graph geometry) and show the dual-space model loses coverage. Without this, the contribution is architectural assertion, not validated research.
5. **Conformal-gauge validation (§12.8).** The "Dirichlet ⇒ minimal surface" chain is currently *imposed*, not *derived* from the discrete wiring pipeline. Empirical validation — spectral stability under subsampling, $\lambda\tau$ tail-quality metrics — must confirm the dual-space diffusion trajectory preserves the same low-frequency geometry a volumetric model would recover.

### 7.3 Open design questions the doc does not resolve

- The topology decoder's MLP width/depth have no design derivation (current defaults `hidden_dim=256`, 3 layers). The decoder is a new construct, so new parameters are expected — but AGENTS.md first principle 2 still requires their *values* to be justified by the design, not copied from generic diffusion decoders. A first-principles derivation: the spectral chart is $K$-dimensional with $K \ll F$ (typically $K = F/4$), so capacity should scale with $K$, not $F$. Proposal: $\text{hidden} = 2K$ (one expansion of the chart), depth 2 (linear–GELU–linear, matching the encoder's two-step structure), giving $\sim 4K^2$ parameters — comparable to the encoder's $\eta, \sigma^2$ budget rather than a $256$-wide $3$-layer net ($\sim 500$k params) that would silently dominate $\mathcal{H}$'s restoring force in the reconstruction loss. The failure mode to avoid: an over-capacity MLP that ignores the geometry §3 builds and reproduces the ambient decoder's collapse under a different name.
- The leapfrog score approximation in `topology_reverse_sde` (using `model._last_Q` as the predicted clean sample) is a modelling choice, not a derived reverse step. A true reverse of the wave recurrence is blocked by the nonlinearity of $B_t$ and $\mathcal{H}$; whether the leapfrog is sufficient is an empirical question the NN-ratio gate answers.
- The determinant drop (§12.7) is rigorous for the graph Dirichlet energy but the conformal-gauge assumption (§12.8) that connects it to a minimal surface is not yet derived from the wiring pipeline. This is the deepest open theoretical item.

---

## 8. Reading order for a new contributor

1. This document (concept + snapshot + roadmap).
2. `docs/00-DESIGN.md` §1, §4, §12 (the full design of the operator and the topology decoder).
3. `docs/001-IMPLEMENTATION-PLAN.md` §5, §6.3, §9 (the implementation contract and the three-sampler hierarchy).
4. `esdm/models/vibrational.py:255` (the encoder in code) and `esdm/models/topology_decoder.py:79` (the decoder in code) — read these two functions side by side; they are the two implementations of $\mathcal{H}$.
5. `esdm/graphs/wiring.py` (ArrowSpace — the geometry both sides share).
6. The issue tracker: #29 (post-v0.2.0 pathway), #24 (2.5-D track), #27 (design validation).

The one idea to carry through all of it: **the semantic operator is the general concept — project onto the ArrowSpace eigenbasis, diffuse along the graph, convert to a Rayleigh-gradient force. The encoder–decoder is the diffusion implementation of that concept — the encoder runs the operator as a restoring force in a vibrational wave recurrence, the decoder runs it backwards as a topology-adaptive sampler on the spectral chart. ESDM is a vibrational model on a graph, not a score-matching model with a graph prior. Making that implementation a trained, balanced, validated pair is the work.**
