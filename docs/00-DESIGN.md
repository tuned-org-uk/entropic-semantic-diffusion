# 00 — Entropic Semantic Diffusion Model (ESDM): Design Document

> **Status:** v0.3.0 — consolidated design (issue #29)
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
| `recon_decrease` | 10 | Reconstruction loss decreases (NOT total ELBO) |
| `net_energy_flow` | 5 | $\Lambda$ vs $\mathcal{H}$ balance ($\approx 0$) |
| `pi_contraction_ratio` | 5 | $\Pi$ activity $\in [0.01, 0.3]$ |

### Diagnostics

Two diagnostics are logged per step:

- `net_energy_flow` $= \|Q_{t+1}\|_F^2 - \|Q_t\|_F^2$ — $\Lambda$ vs $\mathcal{H}$ balance
- `pi_contraction_ratio` $= \|\Delta Q_t\| / \|Q_t\|$ — $\Pi$ activity

| `net_energy_flow` | Interpretation |
|---|---|
| $\approx 0$ | Balanced: $\Lambda$ and $\mathcal{H}$ are in equilibrium |
| $> 0.1 \cdot \|Q_0\|_F^2$ | Forcing dominates: $\mathcal{H}$ is not pulling back fast enough |
| $< -0.5 \cdot \|Q_0\|_F^2$ | Contraction dominates: modes collapsing, generation diversity lost |

### Diversity / quality criteria (for generative evaluation)

| Metric | Definition |
|---|---|
| `coverage` | Fraction of reference samples with a generated neighbour within Chamfer threshold $\delta$ |
| `precision` | Fraction of generated samples within Chamfer threshold $\delta$ of any reference sample |
| `mode_separation` | $\text{std}(R(f_i))$ across generated samples — spectral mode diversity |

An improvement in `precision` without `coverage` is not a win for the dual-space hypothesis. The hypothesis predicts improved basin coverage (higher `coverage`, higher `mode_separation`), not merely sharper samples.

---

## 12. The 2.5-D Dual-Space Track

The 2.5-D track tests whether a dual-space model (2D item coordinate + graph-Laplacian feature signal) can match full 3D volumetric diffusion without computing Jacobian determinants.

### Why the determinant can be dropped

The determinant/Jacobian machinery in continuous Fokker–Planck exists to track how *volume* changes under a flow map. But the graph wiring formalism replaces the ambient manifold with a discrete Laplacian $L_F$ whose Rayleigh quotient $R(x) = x^\top L_F x / x^\top x$ converges to the continuum Dirichlet energy. Because Dirichlet energy is a quadratic form on the graph rather than a measure of local volume distortion, tracking its evolution under noise never requires a determinant — only the Laplacian's spectrum and how mass redistributes across eigenmodes.

### Graph Fokker–Planck

The discrete analogue replaces the continuum Laplacian $\Delta$ with the graph Laplacian $L_F$:

$$p_{t+\Delta t} \approx p_t - \Delta t \, \nabla_{\text{item}}\cdot(p_t v_t) + \frac{\sigma^2\Delta t}{2}\, L_F p_t$$

Here $L_F p_t$ is a sparse matrix–vector product (no determinant, no volume element). This is the "2.5-D" trick: item-space transport (the divergence term, low-dimensional) plus feature-space diffusion (the Laplacian term, driven by the wired graph's spectrum).

### Score as Rayleigh gradient

The score function is redefined as the gradient of the $\lambda\tau$ energy rather than the log-density gradient in ambient space:

$$s(x) \approx -\nabla_x R(x)$$

since minimizing Rayleigh energy is variationally equivalent to following the score toward smoother (lower-Dirichlet-energy) configurations on the wired manifold. This keeps the entire generative trajectory inside the dual item/feature graph representation, using only sparse Laplacian operators — never requiring a 3D volume embedding.

### Update equations

The 2.5-D dual-space update splits the state into a low-dimensional item-coordinate $z_t$ (≤2D) plus a feature-channel signal $f_t$ living on the wired graph:

$$z_{t-\Delta t} = z_t + \eta\,\nabla_z R(f_t)\,\Delta t + \sigma\sqrt{\Delta t}\,\epsilon_z$$

$$f_{t-\Delta t} = f_t - \frac{\sigma^2\Delta t}{2}\, L_F f_t$$

The key structural claim: $(z_t, f_t)$ jointly carries the same generative information as $x_t \in \mathbb{R}^3$.

### Conformal gauge assumption

The "Dirichlet ⇒ minimal surface" chain is rigorous only under conformal-gauge assumptions, which are currently *imposed* rather than *derived* from the discrete wiring pipeline. Empirical validation (spectral stability under subsampling, $\lambda\tau$ tail-quality metrics) must confirm that the dual-space diffusion trajectory preserves the same low-frequency geometry that a volumetric diffusion model would recover.

---

## References

- Barontini, R. (2026). *Testing the problem of time with cold atoms*. Phys. Rev. Research 8, L022047. https://journals.aps.org/prresearch/abstract/10.1103/1h9j-df4k
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*. tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
- Moriondo, L. (2024). *Graph Wiring*. tuned.org.uk/assets/graph_wiring.pdf
- VDT repo: https://github.com/tuned-org-uk/spectral-transport-diff
- VDT Zenodo: https://zenodo.org/records/20816835
- Arxiv paper: https://arxiv.org/abs/2606.21535
