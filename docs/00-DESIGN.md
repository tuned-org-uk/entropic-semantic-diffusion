# 00 — Entropic Semantic Diffusion Model (ESDM): Design Document

> **Status:** v2.0.0 — free wave amplitudes and bounded reconstruction gains.
> The following contract supersedes the unit-sphere projection and identity
> readout in the historical sections below for the entropic-clock ESDM.

### v2 reconstruction contract

#### Proposed v2.0.1 encoder initialisation

The posterior mean should initially express coordinates in the same frozen
chart used by the reconstruction operator:

$$\mu(x)=xU_q,\qquad W_\mu^{(0)}=U_q^\top,\qquad b_\mu^{(0)}=0.$$

The mean map remains trainable. This derives its initial state from the
existing graph without adding a parameter or constraining latent amplitudes.
The log-variance head and Gaussian reparameterisation are unchanged. A saved
checkpoint replaces these initial weights normally. This proposal is tested
under the frozen [autoresearch protocol](../autoresearch.md); acceptance
requires a fresh confirmation set and regression guards. On smooth graph
signals the prior is intentionally informative; its benefit is not evidence
that training discovers the graph geometry.

#### Signal readout

The Gaussian posterior still controls the distribution of latent samples.
Neither those samples nor the raw wave state are clipped or projected to unit
length. Damping, the step-size limit, and the entropy clock govern evolution;
they do not impose a hard amplitude bound. Input preprocessing can still
normalise observed data. Only the **modal gains of the signal readout** are
constrained. With a frozen orthonormal chart (row-vector convention):

$$D_w(Q)=Q+((QU_q)\odot(w-1))U_q^\top,\qquad
\hat x=\sqrt{\bar\alpha}\,D_w(Q)+\sqrt{1-\bar\alpha}\,\epsilon.$$

The complement is transmitted with gain one. For an eigenmode, multiplying
the input amplitude by ten multiplies its readout by ten. The guarantee is on
the multiplier, not the reconstructed feature, the additive Gaussian noise,
the spectral-loading network's trainable matrices, or the whole model.

For ascending retained chart indices use rounded, log-spaced integers from
4096 to 65536 inclusive (one mode uses 4096). This fixed benchmark grid starts
above Schoenfeld's conditional threshold of 2657 and spans four doublings.
These are engineering choices, not graph frequencies. Define the **unoffset**
logarithmic integral by the real exponential integral. Count primes inclusively,
not prime powers. Set

$$\operatorname{li}(x)=\operatorname{Ei}(\log x),\quad
B(x)=\sqrt{x}\log(x)/(8\pi),\quad d_k=B(x_k)/\operatorname{li}(x_k),
\quad w_k=1+d_k\tanh\theta_k.$$

Under classical RH, [Schoenfeld's bound (1976), Corollary 1](https://doi.org/10.1090/S0025-5718-1976-0457374-X)
controls the difference between the prime count and this integral on the chosen domain. It does not bound a
graph waveform. Initialise the learnable logits using the finite measured
residual, so the initial gain is the ratio of the count to the integral:

$$\theta_k^{(0)}=\operatorname{atanh}
 \big((\pi(x_k)-\operatorname{li}(x_k))/B(x_k)\big).$$

Verify the open interval numerically before initialisation, rather than clipping
an invalid residual. The weight bound then holds by construction throughout
optimisation independently of RH. The mode ordering refers to the particular
frozen checkpoint basis, not a rotation-invariant rule on degenerate eigenspaces.
There is one trainable gain per retained mode so fitting can alter the relative
contribution of different directions without altering their amplitudes upstream.

On the full set of 61,441 integers in the grid's range, an inclusive sieve
and float64 evaluation of the integral give a maximum absolute residual
divided by the bound of 0.825800 (at 5380). This finite-domain numerical
check explains why all grid initialisations have finite logits; it does
not establish an asymptotic bound. Unlike the ALD-SC prime-power filter,
this readout uses the ordinary count of primes and learns the gains.

> **Sources:** Barontini 2026 (Phys. Rev. Research 8, L022047), VDT repo, quantum-vibrational repo, spectral-diffusion repo, ArrowSpace graph wiring formalism (Moriondo 2024)
> **Supersedes:** `00-DESIGN_PREAMBLE.md`, `00-DESIGN_HYBRID.md` (now in `docs/legacy/`)

---

## 1. The Two Ideas

The ESDM is built on the fusion of two ideas that, together, replace every external time parameter in a diffusion model with internally derived dynamics. Each idea is independently grounded; their interaction is the architectural contribution.

### 1.1 The Semantic Operator

Every vector space is simultaneously a geometric item-space and a semantic feature-space — the dual-space hypothesis (Moriondo, *ArrowSpace: Semantic Basins*, 2024). In the ESDM this duality is operationalised as a three-step operator $\mathcal{H}$ that fuses a frozen geometric prior with a learned semantic structure:

- **Geometric prior** ($L_F$, $U_q$): the ArrowSpace feature-space Laplacian, built once before training from graph wiring. Its leading eigenpairs $(U_q, \Lambda_q)$ encode the spectral geometry of the semantic field — which feature directions are smooth, which are rough. This is the structural knowledge that defines what coherent semantic modes look like. It is frozen, not learned.

- **Learned semantic structure** ($\sigma^2$, $\eta$, $B_t$): what the model learns to *do* with that geometric prior during training. The Rayleigh-gradient restoring force ($\eta$), the diffusion coefficient ($\sigma^2$), and the attention forcing ($B_t$) are the parameters the optimizer adjusts to shape the dynamics on top of the frozen geometry.

The operator is:

$$\mathcal{H}(Q_t) = \Pi\!\left(\text{Diffuse}_{L_F}\!\left(\Phi(Q_t)\right)\right)$$

where $\Phi$ lifts the item-space state into the feature-space eigenbasis, $\text{Diffuse}_{L_F}$ applies a Laplacian heat step along the geometric graph, and $\Pi$ converts the smoothed signal back into a Rayleigh-gradient restoring force on the item-space state. This is the mechanism by which the dark sector's spectral structure directs the bright sector's diffusion — the semantic operator that gives the model its name.

The geometric prior provides the valid initialisation domain for generation: noise must be constrained to the $U_q$ subspace, not drawn from the full $\mathbb{R}^F$. The learned structure provides the dynamics that, during training, shape the item-space trajectories toward semantic basins.

### 1.2 The Barontini Entropy Clock

Barontini (2026) solves the Wheeler-DeWitt problem of time experimentally: a closed system described by a time-independent Hamiltonian has no built-in mechanism to sequence its own evolution. Using a Bose-Einstein condensate partitioned into a "dark" (unobserved) and "bright" (observed) sector, he shows that an *entropic time* constructed entirely from internal entropy exchange between the sectors can replace the external laboratory clock and reproduce the full measured dynamics via an effective Schrödinger equation parametrised by that internal time.

The ESDM applies this directly. The entropic time is:

$$\tau_n = \sum_{i=1}^{n} |\Delta \,\mathit{kl\_tau}_i|$$

where $\Delta\mathit{kl\_tau}_i$ is the per-step change in the bright-sector KL. This is monotonically increasing by construction — the arrow of time cannot reverse. It stops when entropy is not flowing (heat death), providing a training termination criterion that is thermodynamic, not a step count.

### 1.3 How They Work Together

The semantic operator and the entropy clock are coupled through the wave recurrence:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2\,\mathcal{H}(Q_t) - \gamma\,\Delta Q + \Lambda(\tau)\cdot\Delta\tau^2\,B_t$$

The clock provides $\Delta\tau$ (when to step) and $\Lambda(\tau)$ (how much forcing to inject). The semantic operator provides $\mathcal{H}(Q_t)$ (the restoring force that pulls $Q_t$ toward smooth modes). The tension between the pump $\Lambda$ (exploration — drives $Q_t$ away from smooth modes) and $\mathcal{H}$ (restoration — pulls $Q_t$ toward smooth modes) is the dynamics of the system. When they are balanced, the system oscillates; when the pump dominates, the Q-state is destroyed; when $\mathcal{H}$ dominates, the system collapses to a single mode.

The clock's cosine schedule $\bar{\alpha}(\tau) = \cos^2(\pi/2 \cdot \tau/\tau_\mathit{max})$ gates the diffusion decoder: at $\tau=0$ (start of training), $\bar{\alpha}=1$ and the decoder reconstructs directly from $Q$; as $\tau$ accumulates, $\bar{\alpha}\to 0$ and the decoder must reconstruct from an increasingly diffused $Q$-state. The semantic operator's job is to keep $Q$ recoverable throughout this trajectory — so that at generation time, reversing the trajectory produces on-manifold embeddings.

**The key design constraint:** the pump force and the $\mathcal{H}$ restoring force must be the same order of magnitude throughout training. If the pump is too strong (as was the case with the original $\lambda_\mathit{clip}=100.0$), it destroys the $Q$-state before the decoder can learn to reconstruct. The derivation of $\lambda_\mathit{clip} = 0.1\,F$ from the $\Lambda/\mathcal{H}$ balance (§4.5) is the design-level fix for this constraint.

### 1.4 The 2.5-D Dual-Space: Topology-Adaptive Decoding

The ArrowSpace Laplacian defines an intrinsic metric on a lower-dimensional, non-Euclidean energy surface. The ambient hypersphere $S^{F-1}$ is the wrong reference geometry for generation — it is the ambient space, not the manifold where the data lives. The decoder must invert points on the ArrowSpace energy surface, not the ambient sphere.

**The heat kernel manifold.** ArrowSpace constructs a graph Laplacian $L \in \mathbb{R}^{F \times F}$ with its energy distribution (the lambdas). The energy dispersion on this graph defines a heat kernel manifold:

$$\mathcal{M}_t = \{ e^{-tL} x \mid x \in \mathbb{R}^F, \|x\|=1 \}$$

This is not 3D, not a sphere, not flat — it is a 2.5-D structure with effective dimension between 2 and 3, governed by the spectral gap $\lambda_2$ and the clustering structure. The effective dimension is approximately:

$$d_\mathit{eff} \approx \frac{\left(\sum_k e^{-\lambda_k}\right)^2}{\sum_k e^{-2\lambda_k}} \approx 2\text{–}4$$

For CIFAR-10 DINO-256 with 15 clusters and mean $\lambda = 0.146$, this gives $d_\mathit{eff} \approx 2\text{–}4$ despite living in $\mathbb{R}^{256}$.

**Spectral coordinates.** Any $Q \in \mathbb{R}^F$ decomposes as $Q = \sum_{k=1}^{K} \hat{q}_k \phi_k + Q_\perp$ where $\phi_k$ are the $K$ lowest-eigenvalue eigenvectors of $L_F$ (the smooth modes, i.e. $U_q$), $\hat{q}_k = \langle Q, \phi_k \rangle$ are spectral coefficients, and $Q_\perp$ is the off-manifold residual. The decoder operates on $\hat{q} \in \mathbb{R}^K$ — a K-dimensional spectral chart. **Sampling in spectral coordinates is sampling on the manifold by construction** — a point drawn from $\hat{q} \sim p(\hat{q})$ and lifted back via $\Phi_K \hat{q}$ always lies on the smooth submanifold. Off-manifold generation becomes geometrically impossible.

**Heat kernel prior.** The natural prior on the spectral chart is $\hat{q} \sim \mathcal{N}(0, e^{-2t L_K})$ where $L_K = \Phi_K^\top L_F \Phi_K$ is the projected (reduced) Laplacian. At $t \to 0$, samples concentrate near data points. At $t \to \infty$, samples spread over the full smooth submanifold. The $\bar{\alpha}(\tau)$ schedule *is* this $t$ parameter — the two are directly coupled.

**Connection to the semantic operator (§1.1).** The semantic operator $\mathcal{H} = \Pi \circ \text{Diffuse}_{L_F} \circ \Phi$ operates in the forward direction: it lifts $Q$ onto the spectral chart ($\Phi$), diffuses along the graph ($\text{Diffuse}_{L_F}$), and projects back as a restoring force ($\Pi$). The topology-adaptive decoder is the **inverse** path: it projects $Q$ onto the spectral chart, samples from the heat kernel prior on that chart, and lifts back to ambient space. Together, the semantic operator and the topology decoder form a complete forward-reverse pair on the ArrowSpace manifold.

**The ArrowSpace lambda distribution.** ArrowSpace is the space defined by two components: the graph Laplacian ($L_F$, the `gl` property — structural connectivity) and the lambda distribution (`lambdas()` method — per-item energy scores in $[0, 1]$, the data's distribution within the graph topology). Low lambda = smooth (on-manifold), high lambda = rough (off-manifold). The empirical spectral prior uses the lambdas as weights to bias sampling toward the smooth regime.

See §12 for the full topology-adaptive decoder architecture, the Fokker–Planck derivation, and the conformal gauge assumption.

### 1.5 The Interplay: How the Five Ingredients Lock Together

The five ingredients — geometric space, ArrowSpace, vibrational dynamics, encoding, and decoding — are not independent modules wired in sequence. They interlock through the 2.5-D dual-space principle. This section states the interplay concisely; the full treatment is in `docs/002-SEMANTIC-OPERATOR.md` §4.

**Geometry ↔ ArrowSpace.** The geometric space is $L_F$ — the `gl` half of ArrowSpace. ArrowSpace adds the lambdas: the per-item energy distribution on that graph. Without the lambdas you have a graph Laplacian (a skeleton); with them you have ArrowSpace (the skeleton *plus* the data's position on it). The lambdas are not a weighting applied to $L_F$ after the fact — they are half the space. $L_F$ defines the landscape; the lambdas say where on the landscape the data lives.

**ArrowSpace ↔ Vibrational.** The vibrational dynamics run on the ArrowSpace geometry, not on ambient $\mathbb{R}^F$. The $\Phi$ step of $\mathcal{H}$ projects onto $U_q$ — the smooth eigenbasis of $L_F$ — before the Laplacian acts. This restricts the wave recurrence to the semantically coherent subspace. The $\Pi$ step converts the Laplacian-diffused signal back into a Rayleigh-gradient restoring force. The lambdas enter the generative side: the heat kernel prior $\mathcal{N}(0, e^{-2t L_K})$ is weighted by the lambdas (low lambda = high variance = smooth modes amplified), biasing sampling toward the semantic basins where real data lives. Per-feature lambdas $(F,)$ are projected onto the $K$-dim spectral chart via $\lambda_K^{(k)} = \sum_f \lambda_f \, U_{f,k}^2$.

**Vibrational ↔ Encoding/Decoding.** The encoder *is* the vibrational block — the wave recurrence with $\mathcal{H}$ as restoring force, shaping $Q_t$ toward smooth modes on $S^{F-1}$. The decoder *is* $\mathcal{H}$ inverted: it projects onto the spectral chart $\hat{q} \in \mathbb{R}^K$, samples from the heat kernel prior, and lifts back to $\mathbb{R}^F$. Both sides use the same $U_q$, the same $L_F$, the same eigenbasis. The encoder runs $\mathcal{H}$ deductively (forward); the decoder runs $\mathcal{H}^{-1}$ constructively (reverse).

**Encoding/Decoding ↔ 2.5-D.** The vibrational state $Q_t: (B, F, 1)$ is already 2.5-D: $F$ feature-nodes each carrying one item-coordinate. The encoder is 2.5-D by construction, not by addition. The decoder completes the pair by operating on the $K$-dim spectral chart — the intrinsic manifold — rather than the ambient $\mathbb{R}^F$. The intentional encode/decode asymmetry (full-$F$ forward, 2.5-$D$ reverse) is a DDIM-style relaxation: the reverse process operates on a reduced chart, not the full ambient space. The `score25d.py` scaffold makes the 2.5-D split explicit (item coordinate $z_t$ + feature signal $f_t$ on the wired graph); the topology decoder is its natural generation side.

$$
\underbrace{L_F,\;U_q}_{\text{geometry}} \;\xrightarrow{\text{+lambdas}}\; \underbrace{\textit{gl}+\textit{lambdas}}_{\text{ArrowSpace}} \;\xrightarrow{\text{wave recurrence}}\; \underbrace{\mathcal{H}\;\text{on}\;S^{F-1}}_{\text{vibrational}} \;\xrightarrow{\text{forward/reverse}}\; \underbrace{\text{encode}\;/\;\text{decode}}_{\text{2.5-D}}
$$

The single chain: ArrowSpace builds $L_F$ and the lambdas from the data corpus; $L_F$'s eigenpairs define $U_q$, the spectral chart; the vibrational block runs $\mathcal{H}$ on that chart as the encoder; the topology decoder runs $\mathcal{H}^{-1}$ on the same chart as the decoder. Every ingredient flows through the 2.5-D dual-space — none is optional, none is bolted on.

---

## 2. The Two-Sector Partition

### 2.1 Physical analogue

Barontini partitions the closed BEC Hamiltonian as:

$$\hat{H} = \hat{H}_\mathit{bright} + \hat{H}_\mathit{dark} + \hat{H}_\mathit{coupling}$$

where `bright` is the observed sector, `dark` the unobserved sector, and `coupling` the optical barrier (height $V$) controlling entropy exchange between them.

### 2.2 Semantic analogue

The ESDM maps this partition onto the VDT/SDM architecture:

| Barontini | ESDM |
|---|---|
| Bright sector (observed BEC half) | Active spectral modes: $\omega_k > \theta$ — the `N_active` population |
| Dark sector (unobserved BEC half) | Latent wiring $L(z)$, inactive modes, ArrowSpace prior eigenpairs $(U_{1:q}, \lambda_{1:q})$ |
| Coupling strength $V$ | `kl_S` regularisation weight — controls entropy exchange between sectors |
| Barrier height → big bang / big crunch | Mode activation / mode collapse events during training |
| BEC atom count $N(\phi)$ in bright sector | `N_active` — number of modes above threshold |
| Clock field $\phi$ (centre of mass) | `log_lam_s` — the learnable log spectral scale |
| Scale factor $a$ (BEC RMS radius) | Reconstruction width $\Sigma$ — spread of the decoded embedding |

The total semantic Hamiltonian is:

$$\hat{H}_\mathit{sem} = \hat{H}_\mathit{item} + \hat{H}_\mathit{spectral} + \hat{H}_\mathit{wiring}(V)$$

### 2.3 Why the partition matters

- **Item sector** (bright) is observable: we can measure $\hat{x}$, compute reconstruction loss, and inspect the active mode population.
- **Spectral sector** (dark) is latent: it governs the dynamics of item-space but is never directly observed during inference.
- Entropy flows *between* sectors even though the total system entropy is conserved — exactly as in Barontini's miniuniverse. The `entropy conservation` term in the ELBO enforces this (§6).

### 2.4 Mode threshold

The threshold $\theta$ separating bright from dark is the **median** of $\omega$ across modes, computed per-batch at each step. This ensures ~half the modes are bright and ~half are dark, matching the Barontini two-sector analogue (equal partition). The threshold is not a parameter — it is a derived quantity.

---

## 3. Entropic Time

### 3.1 Definition

Barontini defines entropic time (Eq. 3 of the paper) as:

$$\tau(\lambda) = \frac{\sigma}{k_B} \int_\lambda \frac{dS}{d\phi} \, |d\phi|$$

This is monotonically increasing by construction: as long as $dS$ and $d\phi$ share sign, the arrow of time cannot reverse. For the ESDM, the clock field $\phi \equiv \log\lambda_S$ is already present in SDM's `log_lam_s`, and the entropy in the bright sector is tracked by `kl_tau`. The discrete entropic time accumulator is:

$$\tau_n = \sum_{i=1}^{n} |\Delta \, \mathit{kl\_tau}_i|$$

### 3.2 Key properties

1. **Monotonic** — the sum of absolute values cannot decrease, so the arrow of time is guaranteed.
2. **Internally derived** — computed entirely from observable quantities (`kl_tau`) with no external clock.
3. **Stops when entropy is not flowing** — if `kl_tau` is stable, $\tau_n$ stops increasing. This is the semantic equivalent of Barontini's observation that "no entropic time elapses between a big crunch and the subsequent big bang, because no entropy is exchanged there."
4. **Heat death detection** — when $\tau_n$ stalls for many consecutive steps, the system has converged to a stationary state. This is a training termination criterion, not an epoch count.

### 3.3 Non-monotonicity resolution

The clock field `log_lam_s` does not evolve monotonically — it can increase and decrease across training. Barontini explicitly identifies this as the non-monotonicity problem for recollapsing minisuperspace models, and resolves it via the absolute-value integral. The ESDM inherits this fix: the entropic time $\tau$ is always increasing even when `log_lam_s` oscillates.

### 3.4 Clock scaling

The entropic clock accumulates **raw** `kl_tau` (not normalised by $S_0$): $\tau_n = \sum |\Delta \mathit{kl\_\tau}|$. Normalising by $S_0$ compressed per-step deltas to ~3e-4 on CIFAR-10 ($S_0$~52), freezing the clock. Raw kl_tau gives ~0.2/step.

`tau_max_est` is derived as $3 \cdot S_0$ at initialisation. With $S_0 \approx 52$ on CIFAR-10, this gives $\tau_\mathit{max} \approx 156$, spreading the cosine schedule across the full run. This is a derived constant, not a parameter.

---

## 4. The Semantic Operator: Hybridisation $\mathcal{H}$

### 4.1 The problem

ESDM operates on two spaces simultaneously:

- **Feature space** (dark sector): $L_F \in \mathbb{R}^{F \times F}$ — the graph Laplacian of the *feature dimensions*, built by ArrowSpace graph wiring. Its leading eigenpairs $(U_q, \Lambda_q)$ encode the spectral geometry of the semantic field.
- **Item space** (bright sector): $Q_t \in \mathbb{R}^{B \times F}$ — one row per item in the batch, evolving through the string/spring wave recurrence.

The term $L_F Q_t$ in the wave recurrence mixes these two spaces: $L_F$ (F×F, feature geometry) acts on $Q_t$ (B×F, item states), and the result is fed back into the item-space dynamics. The hybridisation operator makes the **lifting** (how an item-state enters the feature space before $L_F$ acts) and the **projection** (how the Laplacian-diffused feature signal returns to the item space) explicit.

### 4.2 Step 1 — Lifting $\Phi$

$$\Phi:\; Q_t \in \mathbb{R}^{B \times F} \;\longrightarrow\; F_t \in \mathbb{R}^{B \times F}$$

$$\boxed{F_t = Q_t \, U_q \, U_q^\top}$$

where $U_q \in \mathbb{R}^{F \times q}$ are the $q$ leading non-trivial eigenvectors of $L_F$ (the ArrowSpace spectral embedding), with $q \leq F$.

$\Phi$ projects each item's vibrational state onto the subspace spanned by the $q$ dominant feature-space modes, then lifts it back to $\mathbb{R}^F$. The result $F_t$ is a **spectrally filtered version of $Q_t$**: components orthogonal to the leading eigenvectors (high-frequency, locally incoherent modes) are suppressed before the Laplacian acts. This restricts the spring/string system to the semantically coherent subspace of feature space.

$U_q$ is produced by `SpectralLoadingDecoder` as part of the spectral loading prior $(U_q, \Lambda_q)$. The lifting step reuses these eigenvectors without introducing new parameters. $U_q$ is frozen after graph build — it is a buffer, not a Parameter.

### 4.3 Step 2 — Feature-space Laplacian diffusion

$$\text{Diffuse}_{L_F}:\; F_t \in \mathbb{R}^{B \times F} \;\longrightarrow\; \tilde{F}_t \in \mathbb{R}^{B \times F}$$

$$\boxed{\tilde{F}_t = F_t - \frac{\sigma^2 \Delta\tau}{2}\,L_F F_t}$$

Each item's lifted feature signal diffuses along the F×F feature-space graph. Items whose feature signals are far from the graph's smooth modes are pulled toward them; items already in a smooth mode are held there.

The diffusion coefficient uses $\sigma^2 \Delta\tau / 2$ (a heat-equation step), not $\Delta\tau^2$ (a wave-equation step). This is deliberate: Step 2 is a **parabolic** (irreversible) smoothing of the feature signal, while the wave recurrence in Step 3 is **hyperbolic** (oscillatory). The hybridisation operator combines both character types: diffusion in feature space, oscillation in item space.

### 4.4 Step 3 — Projection $\Pi$ (Rayleigh gradient)

$$\Pi:\; \tilde{F}_t \in \mathbb{R}^{B \times F} \;\longrightarrow\; \Delta Q_t \in \mathbb{R}^{B \times F}$$

$$\boxed{\Delta Q_t = -\eta \,\nabla_{Q_t} R(\tilde{F}_t)}$$

where the Rayleigh gradient is:

$$\nabla_{Q_t} R(\tilde{F}_t) = \frac{2}{{\|\tilde{F}_t\|^2}}\left(L_F \tilde{F}_t - R(\tilde{F}_t)\,\tilde{F}_t\right) U_q^\top$$

with $R(\tilde{F}_t) = \tilde{F}_t^\top L_F \tilde{F}_t / \tilde{F}_t^\top \tilde{F}_t$ the Rayleigh quotient (ArrowSpace taumode score), computed row-wise across the batch, and $\eta > 0$ a step size derived from the spectral scale of $L_F$.

**Sign convention in code.** `_hybridisation()` returns `H_action = eta * grad_R` (analogous to `L*Q`). The wave recurrence applies $-dt^2 \cdot \text{H\_action}$, giving the restoring force $-\eta \Delta\tau^2 \nabla R$ toward low-energy modes. The design-doc sign $\Delta Q_t = -\eta \nabla R$ and the code are consistent when read together with the $-dt^2$ factor in `_step()`.

$\Pi$ converts the diffused feature signal back into a **gradient force** on the item-space state $Q_t$. Items whose feature signals have high Rayleigh quotient (far from smooth modes, high spectral energy) experience a larger restoring force pulling them toward lower-energy regions of the feature-space graph. Items already in a smooth mode experience near-zero force. This is precisely the string/spring restoring force: the Rayleigh quotient measures how far a string is stretched relative to the graph's natural modes, and $-\nabla R$ is the force that relaxes it.

### 4.5 Full hybridised wave recurrence

Substituting all three steps into the wave recurrence:

$$\boxed{
Q_{t+1} = 2Q_t - Q_{t-1}
\;-\; \Delta\tau^2\,\mathcal{H}(Q_t)
\;-\; \gamma\,\Delta Q
\;+\; \Lambda(\tau)\cdot\Delta\tau^2\,B_t
}$$

Expanding:

$$Q_{t+1} = 2Q_t - Q_{t-1}
\;+\; \eta\,\Delta\tau^2\,\nabla_{Q_t} R\!\left(
  Q_t U_q U_q^\top
  - \tfrac{\sigma^2\Delta\tau}{2}\,L_F (Q_t U_q U_q^\top)
\right)
\;-\; \gamma\,\Delta Q
\;+\; \Lambda(\tau)\cdot\Delta\tau^2\,B_t$$

The three operators act in sequence:
1. $\Phi$ — restrict to semantically coherent subspace (spectral projection)
2. $\text{Diffuse}_{L_F}$ — smooth along feature-space graph (Laplacian heat flow)
3. $\Pi$ — convert smoothed signal back to item-space restoring force (Rayleigh gradient)

**Geometric constraint: $S^{F-1}$ manifold.** The input embeddings are L2-normalised ($\|x\| = 1$ per sample), so the ESDM state lives on the unit hypersphere $S^{F-1}$. The wave recurrence must preserve this constraint: after each recurrence step, $Q$ is projected back onto $S^{F-1}$ via L2 normalisation. Without this, $Q$ drifts off the manifold and the Rayleigh gradient in $\mathcal{H}$ shrinks (it is normalised by $\|\tilde{F}_t\|^2$), causing CFL instability — the wave recurrence becomes an energy amplifier rather than a bounded oscillator. The normalisation also ensures $\mathit{pcr} = \|\Delta Q_t\| / \|Q_t\|$ measures the true Rayleigh gradient magnitude (since $\|Q_t\| = 1$ always), and the diagnostic for energy flow becomes angular displacement on the sphere rather than norm-squared difference.

### 4.6 The $\Lambda/\mathcal{H}$ balance and $\lambda_\mathit{clip}$ derivation

The pump $\Lambda(\tau) = |\partial_\phi \mathit{kl\_tau}|^{-1}$ diverges when the entropy derivative vanishes (stable modes at init). The clip $\lambda_\mathit{clip}$ bounds this divergence. The correct value is derived from the balance between the pump force and the $\mathcal{H}$ restoring force at the clip point:

$$\underbrace{\lambda_\mathit{clip} \cdot \Delta\tau^2 \cdot \|B_t\|}_{\text{pump force at clip}} \;\sim\; \underbrace{\eta \cdot \Delta\tau^2 \cdot \|\nabla R\|}_{\mathcal{H} \text{ restoring force}}$$

With L2-normalised data ($\|Q\| \approx 1$ per sample), $\|B_t\| \approx 1$ (residual network), and $\|\nabla R\| \approx 2\lambda_\mathit{max}/\|Q\|^2 = 2\lambda_\mathit{max}$ (worst-case Rayleigh gradient), substituting $\eta = 0.1\,F / (2\lambda_\mathit{max})$:

$$\lambda_\mathit{clip} \;\approx\; \eta \cdot 2\lambda_\mathit{max} \;=\; \frac{0.1\,F}{2\lambda_\mathit{max}} \cdot 2\lambda_\mathit{max} \;=\; 0.1\,F$$

This follows the same derivation pattern as $\eta$ (same $0.1$ factor, same $F$). For CIFAR-10 ($F=256$): $\lambda_\mathit{clip} = 25.6$. The previous hardcoded value of $100.0$ was $4\times$ too high, causing the pump to dominate $\mathcal{H}$ at init and destroy the $Q$-state before the decoder could learn to reconstruct.

### 4.7 Architectural diagram

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
```

### 4.8 Degenerate and special cases

| Condition | Effect |
|---|---|
| $q = F$ | $\Phi = \text{id}$, no projection → recovers the bare $L_F Q_t$ formulation |
| $\sigma^2 \to 0$ | Diffuse step vanishes → $\tilde{F}_t = F_t$ → hybridisation reduces to pure Rayleigh-gradient restoring force on the projected state |
| $\eta \to 0$ | Projection step vanishes → feature-space diffusion has no effect on item dynamics → dark and bright sectors decouple |
| $q = 1$ | Only the Fiedler vector is used → maximum compression; item dynamics governed entirely by the graph's primary connectivity axis |

---

## 5. The Entropic Pump and Collocation

### 5.1 The entropic Schrödinger equation

Starting from the static WDW equation and applying a Feshbach-Villars decomposition (keeping only forward-in-time solutions), Barontini obtains (Eq. 6):

$$i\hbar \,\partial_\tau \psi(\tau, a) = \Phi(\tau)\psi(\tau, a) + \Lambda(\tau) \, H_\mathit{geom} \, \psi(\tau, a)$$

where $\Phi(\tau) = \alpha\omega\phi^2 d_\tau\phi$ is a global phase and $\Lambda(\tau) = (\partial_\phi S)^{-1} k_B / \sigma\omega\phi$ is the **entropic pump** — an entropy-dependent energy amplifier. When $\partial_\tau \Lambda = 0$ (no entropy flow), this reduces to an ordinary Schrödinger equation.

### 5.2 The discrete entropic pump

In the ESDM:

- $\Lambda(\tau) = |\partial_\phi \mathit{kl\_tau}|^{-1}$ — the discrete entropic pump
- $\Delta\tau = |\Delta \mathit{kl\_tau}_t|$ — the entropic time step at step $t$

**Interpretation of $\Lambda(\tau)$ as pump:**
- When $\Lambda$ is large (entropy derivative small — modes stable), attention forcing is amplified. The system is in a low-entropy-flow regime and the forcing drives exploration.
- When $\Lambda$ is small (entropy derivative large — rapid mode transitions), attention forcing is suppressed. The system is in a high-entropy-flow regime and the wave recurrence runs close to free vibrational dynamics.
- When $\partial_\tau \Lambda < 0$ (pump shrinking), energy is flowing *into* the bright sector — an expansion phase.
- When $\partial_\tau \Lambda > 0$ (pump growing), energy is being extracted from the bright sector — a contraction phase heading toward "big crunch".

### 5.3 The entropic clock as principled collocation strategy

Physics-informed neural networks (PINNs) are typically analysed as three separate design choices: architecture (what the model can represent), loss function (what the model is pushed toward), and **collocation strategy** (where in the domain the physics residual is evaluated). The ESDM differs from standard PINNs in that the physics is not an external residual penalty — it *is* the forward pass. The wave recurrence is executed, not approximated. The entropic clock provides a principled collocation answer through three mechanisms:

1. **Temporal silencing.** When $\Delta\tau \approx 0$, the recurrence term $\Delta\tau^2\,\mathcal{H}(Q_t)$ vanishes algebraically. The wave equation is not evaluated — it is silenced. The clock tells you *where not* to evaluate the physics residual.

2. **Amplitude modulation via $\Lambda(\tau)$.** The entropic pump scales the stochastic forcing term $B_t$. High $\Lambda$ means the injection is large relative to the restoring force, pushing the system into exploratory regimes. Low $\Lambda$ means the system runs close to free vibrational dynamics. The clock determines *how strongly* to evaluate the physics at each collocation point.

3. **Schedule-adaptive reconstruction.** The cosine schedule $\bar{\alpha}(\tau)$ determines what the diffusion decoder must reconstruct at each $\tau$. Early in training ($\bar{\alpha}\approx 1$), the decoder sees nearly clean $Q$ — easy reconstruction. Late in training ($\bar{\alpha}\approx 0$), the decoder sees fully diffused $Q$ — hard reconstruction. The clock sequences the difficulty schedule automatically.

---

## 6. The ELBO

The ESDM extends the VDT three-term ELBO with an entropy conservation term:

$$\mathcal{L}_\mathit{ESDM} = \underbrace{\mathbb{E}_q[\log p(x \mid z, W)]}_\mathit{reconstruction} - \underbrace{\mathit{KL}(q(z) \| \mathcal{N}(0,I))}_\mathit{kl\_z} - \underbrace{\mathit{KL}(q(S) \| p(S|I))}_\mathit{kl\_S} - \underbrace{\mathit{KL}(q(\omega) \| p(\omega|\tau,L))}_\mathit{kl\_tau} - \underbrace{\beta \cdot |S_\mathit{bright}(\tau) + S_\mathit{dark}(\tau) - S_0|^2}_\mathit{entropy\ conservation}$$

where $\beta = 1/S_0$ by default (dimensionless). $S_\mathit{dark}(\tau)$ is the KL of inactive modes; $S_\mathit{bright}(\tau)$ is `kl_tau` of active modes; $S_0$ is their sum at initialisation, frozen as a buffer.

The conservation term penalises violation of the conservation law that Barontini verifies experimentally — total entropy of the closed system must remain constant even as entropy flows between sectors.

**Note on the conservation tautology.** In the current implementation, `dark_entropy = max(0, S_0 - kl_S - kl_tau)` structurally enforces conservation by construction. This means `conserved_ratio = 0` is a mathematical identity, not a measurement. The real conservation check is `net_energy_flow ≈ 0` (see §11).

---

## 7. The Entropic Diffusion Schedule

The fixed SDM `cosine_alpha_bar(t, T=200)` is replaced by:

$$\bar{\alpha}(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau}{\tau_\mathit{max}}\right)$$

where $\tau$ is accumulated entropic time, not a fixed step count. This means:

- Steps where `kl_tau` is changing rapidly get *more* diffusion time (larger $\Delta\tau$), dissolving features at semantically active moments.
- Steps where entropy is flat get *less* time — the schedule slows down at plateaus automatically.
- $\tau_\mathit{max}$ is determined empirically as $3 \cdot S_0$, not preset.

The `EntropicDiffusionDecoder` computes:

$$\hat{x} = \sqrt{\bar{\alpha}(\tau)} \cdot Q + \sqrt{1 - \bar{\alpha}(\tau)} \cdot \epsilon$$

With $\epsilon = 0$ (deterministic reconstruction), $\hat{x} = \sqrt{\bar{\alpha}} \cdot Q$. At init, $\bar{\alpha} \approx 1$ so $\hat{x} \approx Q \approx x$. As $\tau$ accumulates, $\bar{\alpha} \to 0$ and $\hat{x} \to 0$.

---

## 8. Architecture

```
Input x (B, F) — L2-normalised embeddings
      │
      ▼
[1. Variational encoder]      — x -> (mu, logvar) -> z
      │
      ▼
[2. SpectralLoadingDecoder]  — z, U_q -> W, omega, S, L_z, log_var_S
      │    (spectral metrics, N_active, kl_S, kl_tau)
      ▼
[3. EntropicClock]           — |Delta kl_tau| -> tau_n, Lambda(tau), alpha_bar(tau)
      │
      ▼
[4. VibrationalStateBlock]   — wave recurrence with H(Q_t) gated by Lambda(tau)
      │    (Q_t, rho_+, rho_-)
      ▼
[5. EntropicDiffusionDecoder] — alpha_bar(tau) * Q -> x_hat
      │    (x_hat)
      ▼
   Prediction / Generation
```

| Module | Source | Role |
|---|---|---|
| Variational encoder | New | $x \to (\mu, \text{logvar}) \to z$ for the spectral decoder |
| `SpectralLoadingDecoder` | VDT | $z, U_q \to W, \omega, S, L_z$ — spectral loading prior, mode threshold, KL terms |
| `EntropicClock` | New | Accumulates $\|\Delta\mathit{kl\_tau}\|$, computes $\Lambda(\tau)$, emits $\bar{\alpha}(\tau)$, detects heat death |
| `VibrationalStateBlock` | VDT + $\mathcal{H}$ | Wave recurrence with hybridisation operator $\mathcal{H}(Q_t)$, gated by $\Lambda(\tau)$ |
| `EntropicDiffusionDecoder` | SDM | $\bar{\alpha}(\tau)$-indexed reconstruction: $\hat{x} = \sqrt{\bar{\alpha}} \cdot Q$ |

### Signed interference as a prediction signal

The quantum-vibrational paper establishes that the amplitude Laplacian $\Lambda_A(t)$ has negative eigenvalues precisely when signed interference is active. In the ESDM this becomes a **prediction confidence signal**: when $\rho_- > \rho_+$ in the `SignedDensityMatrix` (more destructive than constructive interference), the model is in a decoherent state and predictions should be flagged as low-confidence. This is structurally equivalent to Barontini's observation that the entropic clock stops near a big crunch — no time flows, no information is being processed, no reliable prediction should be made.

---

## 9. KL Terms and Their Roles

| Term | Role in ESDM | What breaks if you drop it? |
|---|---|---|
| `recon` | Reconstruction loss — decoder must invert $Q \to \hat{x}$ | No learning signal; model outputs constants |
| `kl_z` | Standard VAE regulariser on $z$; keeps latent sane | Latent blows up / overfits; clock still works |
| `kl_S` | Controls coupling between items and feature Laplacian; drives barrier sweep | Two-sector "mini-universe" analogy collapses; barrier-height experiment meaningless |
| `kl_tau` | Defines bright-sector entropy; **this is the clock's fuel** | Entropic time $\tau_n$ becomes undefined; core design collapses |
| Conservation | Ensures $S_\mathit{bright} + S_\mathit{dark} \approx S_0$; matches Barontini's conserved miniuniverse | Dark/bright split loses physical meaning; clock still runs but isn't "two-sector" |

**`kl_tau` is non-negotiable.** Without it, $\tau_n = \sum |\Delta\mathit{kl\_tau}|$ doesn't exist. `kl_S` and the conservation term are conceptually important but experimentally optional — they can be tuned down to test behaviour. `kl_z` is standard VAE hygiene; it can be annealed during early experiments.

---

## 10. Architecture Constants

These values are **constants**, not hyperparameters. They are derived from the design and must not be swept, parameterised, or overridden without a design-doc amendment.

| Constant | Value | Source |
|---|---|---|
| $q$ (spectral modes) | $F/4$ (rounded to nearest power of 2) | §4.2, default |
| $\gamma$ (damping) | 0.1 | VDT default |
| `n_vibrational_steps` | 2 | CFL-safe for real $L_F$ |
| $\lambda_\mathit{clip}$ | derived: $0.1\,F$ | §4.6 ($\Lambda/\mathcal{H}$ balance derivation) |
| $\eta$ (Rayleigh step size) | derived: $0.1\,F / (2\lambda_\mathit{max})$ | §4.4 (spectral scale of $L_F$) |
| $\sigma^2$ (diffusion coefficient) | 1.0 | §4.3 |
| `grad_clip` | 1.0 | Training stability |
| `seed` | 3407 | Project-wide |
| `batch_size` | 64 | Standard |
| `k` (ArrowSpace clusters) | 4 | Graph construction |
| `mode_threshold` | median($\omega$) per-batch | §2.4 |
| `tau_max_est` | $3 \cdot S_0$ (derived) | §3.4 |
| `eps_rel` | dataset-specific (e.g. 0.002 for CIFAR-10) | Per-dataset YAML config |
| `heat_death_patience` | dataset-specific (e.g. 50 for CIFAR-10) | Per-dataset YAML config |

The only free hyperparameters are `lr` and `n_steps` (governed by heat death). Dataset-specific constants (`eps_rel`, `heat_death_patience`, `default_lr`, `graph_params`) come from per-dataset YAML configs.

---

## 11. Fitness Criteria

The composite fitness score (higher is better; maximum 85) is computed from a single training run. It measures health criteria from the design's physics.

| Criterion | Max | What it measures |
|---|---|---|
| `tau_monotone` | 10 | $\tau_n$ strictly increasing |
| `tau_range_bonus` | 5 | Total $\tau_n$ accumulation (larger range = better) |
| `alpha_bar_decrease` | 10 | $\bar{\alpha}(\tau)$ drops from ~1 toward ~0 |
| `conservation` | 15 | $\|S_\mathit{total} - S_0\| / S_0 \to 0$ (currently tautological — enforced by construction) |
| `heat_death` | 10 | Heat death fires within the run |
| `heat_death_timing` | 5 | Fires around 70% through (not too early/late) |
| `rho_sign_stabilizing` | 10 | $\rho_-/\rho_+$ variance decreasing |
| `recon_decrease` | 10 | Median reconstruction loss during active phase ($\alpha_\mathit{bar} > 0.01$) |
| `angular_displacement` | 5 | Angular displacement of $Q$ on $S^{F-1}$: $\arccos(Q_{t+1} \cdot Q_t)$ |
| `pi_contraction_ratio` | 5 | $\Pi$ activity $\in [0.3, 3.5]$ (recalibrated for $\|Q\|=1$) |

### Diagnostics

Two diagnostics are logged per step:

- `angular_displacement` $= \arccos(Q_{t+1} \cdot Q_t)$ — how much $Q$ moves on $S^{F-1}$ per step. Replaces `net_energy_flow` (which was identically 0 with $S^{F-1}$ normalisation). Healthy range $[0.01, 0.5]$ radians.
- `pi_contraction_ratio` $= \|\Delta Q_t\| / \|Q_t\|$ — $\Pi$ activity. With $\|Q\|=1$, this directly measures the Rayleigh gradient magnitude. Healthy range $[0.3, 3.5]$ (IQR-calibrated for the $S^{F-1}$ regime).
- `gamma_mean` — mean damping coefficient. Monitored to detect gamma collapse (learned to zero).

### Diversity / quality criteria (for generative evaluation)

| Metric | Definition |
|---|---|
| `coverage` | Fraction of reference samples with a generated neighbour within Chamfer threshold $\delta$ |
| `precision` | Fraction of generated samples within Chamfer threshold $\delta$ of any reference sample |
| `mode_separation` | $\text{std}(R(f_i))$ across generated samples — spectral mode diversity |

An improvement in `precision` without `coverage` is not a win for the dual-space hypothesis. The hypothesis predicts improved basin coverage (higher `coverage`, higher `mode_separation`), not merely sharper samples.

---

## 12. The Topology-Adaptive Decoder: Implementation Details

The conceptual foundation is in §1.4. This section provides the implementation details: the decoder architecture, the Fokker–Planck derivation, and the conformal gauge assumption.

### 12.1 The heat kernel manifold (expansion of §1.4)

As stated in §1.4, the energy dispersion on the graph Laplacian defines a heat kernel manifold:

$$\mathcal{M}_t = \{ e^{-tL} x \mid x \in \mathbb{R}^F, \|x\|=1 \}$$

This is not 3D, not a sphere, not flat — it is a 2.5-D structure with fractal-like effective dimension between 2 and 3, governed by the spectral gap $\lambda_2$ of $L$ and the clustering structure. The Laplacian eigenvectors span a smooth low-dimensional surface embedded in $\mathbb{R}^F$, but its intrinsic dimension is far lower than $F$ and non-integer due to multi-scale clustering.

The effective dimension is approximately:

$$d_\mathit{eff} \approx \frac{\left(\sum_k e^{-\lambda_k}\right)^2}{\sum_k e^{-2\lambda_k}} \approx 2\text{–}4$$

### 12.2 Spectral coordinates (expansion of §1.4)

Any $Q \in \mathbb{R}^F$ decomposes as:

$$Q = \sum_{k=1}^{K} \hat{q}_k \phi_k + Q_\perp$$

where $\phi_k$ are the $K$ lowest-eigenvalue eigenvectors of $L_F$ (the smooth modes, i.e. $U_q$), $\hat{q}_k = \langle Q, \phi_k \rangle$ are spectral coefficients, and $Q_\perp$ is the off-manifold residual. The decoder operates on $\hat{q} \in \mathbb{R}^K$ — a K-dimensional spectral chart — rather than the full $F$-dim ambient $Q$.

**Sampling in spectral coordinates is sampling on the manifold by construction.** A point drawn from $\hat{q} \sim p(\hat{q})$ and lifted back via $\Phi_K \hat{q}$ always lies on the smooth submanifold. Off-manifold generation becomes geometrically impossible — this is the key architectural insight that dissolves the NN ratio problem.

### 12.3 Heat kernel as the natural prior

The natural prior on the spectral chart is the heat kernel:

$$\hat{q} \sim \mathcal{N}(0, e^{-2t L_K})$$

where $L_K = \Phi_K^\top L_F \Phi_K \in \mathbb{R}^{K \times K}$ is the projected (reduced) Laplacian. At $t \to 0$, samples concentrate near data points. At $t \to \infty$, samples spread over the full smooth submanifold. The $\bar{\alpha}(\tau)$ schedule *is* this $t$ parameter — the two are directly coupled.

### 12.4 Topology-adaptive decoder architecture

The decoder path replaces the ambient-space decoder with one that respects the graph topology as a geometric prior:

```
Q ∈ R^F
  │
  ▼  project onto Φ_K (smooth eigenvectors = U_q)
q̂ ∈ R^K          ← intrinsic manifold coordinates (spectral chart)
  │
  ▼  heat_kernel_sample(L_K, α_bar(τ))
q̂_noisy ∈ R^K   ← manifold-aware noised sample
  │
  ▼  TopologyDecoder MLP
x̂ ∈ R^F          ← reconstructed embedding
```

This is the implementation of the semantic operator's geometry: the frozen geometric prior ($U_q$, $L_F$) defines the manifold, the learned semantic structure ($\sigma^2$, $\eta$, $B_t$) shapes the dynamics on it, and the topology decoder inverts points on the manifold rather than the ambient space.

### 12.5 Connection to the semantic operator (§1)

The semantic operator $\mathcal{H} = \Pi \circ \text{Diffuse}_{L_F} \circ \Phi$ operates in the forward direction: it lifts $Q$ onto the spectral chart ($\Phi$), diffuses along the graph ($\text{Diffuse}_{L_F}$), and projects back as a restoring force ($\Pi$). The topology-adaptive decoder is the **inverse** path: it projects $Q$ onto the spectral chart, samples from the heat kernel prior on that chart, and lifts back to ambient space. Together, the semantic operator and the topology decoder form a complete forward-reverse pair on the ArrowSpace manifold.

### 12.6 The ArrowSpace lambda distribution

ArrowSpace is the space defined by two components:
- **Graph Laplacian** ($L_F$, the `gl` property): structural connectivity, eigenvectors define smooth modes
- **Lambda distribution** (`lambdas()` method): per-item energy scores in $[0, 1]$, the distribution of the data within the graph topology

The lambdas encode where each item sits in the energy landscape: low lambda = smooth (on-manifold), high lambda = rough (off-manifold). The empirical spectral prior uses the lambdas as weights (low lambda = high weight) to bias the prior toward the smooth regime where real data lives.

### 12.7 Why the determinant can be dropped

The determinant/Jacobian machinery in continuous Fokker–Planck exists to track how *volume* changes under a flow map. But the graph wiring formalism replaces the ambient manifold with a discrete Laplacian $L_F$ whose Rayleigh quotient $R(x) = x^\top L_F x / x^\top x$ converges to the continuum Dirichlet energy. Because Dirichlet energy is a quadratic form on the graph rather than a measure of local volume distortion, tracking its evolution under noise never requires a determinant — only the Laplacian's spectrum and how mass redistributes across eigenmodes.

The score function is redefined as the gradient of the $\lambda\tau$ energy rather than the log-density gradient in ambient space:

$$s(x) \approx -\nabla_x R(x)$$

since minimizing Rayleigh energy is variationally equivalent to following the score toward smoother (lower-Dirichlet-energy) configurations on the wired manifold.

### 12.8 Conformal gauge assumption

The "Dirichlet ⇒ minimal surface" chain is rigorous only under conformal-gauge assumptions, which are currently *imposed* rather than *derived* from the discrete wiring pipeline. Empirical validation (spectral stability under subsampling, $\lambda\tau$ tail-quality metrics) must confirm that the dual-space diffusion trajectory preserves the same low-frequency geometry that a volumetric diffusion model would recover.

---

## References

- Barontini, R. (2026). *Testing the problem of time with cold atoms*. Phys. Rev. Research 8, L022047. https://journals.aps.org/prresearch/abstract/10.1103/1h9j-df4k
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*. tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
- Moriondo, L. (2024). *Graph Wiring*. tuned.org.uk/assets/graph_wiring.pdf
- VDT repo: https://github.com/tuned-org-uk/spectral-transport-diff
- VDT Zenodo: https://zenodo.org/records/20816835
- Arxiv paper: https://arxiv.org/abs/2606.21535
