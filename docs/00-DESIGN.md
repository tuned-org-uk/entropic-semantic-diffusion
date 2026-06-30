# 00 — Entropic Semantic Diffusion Model (ESDM): Design Document

> **Status:** Conceptual design — pre-implementation  
> **Sources:** Barontini 2026 (Phys. Rev. Research 8, L022047), VDT repo, quantum-vibrational repo, spectral-diffusion repo  
> **Author:** Research note synthesised from prior work on Vibrational Deductive Transformer (VDT) and Spectral Diffusion Model (SDM)

---

## 1. The Core Problem

The VDT wave recurrence, the SDM diffusion schedule, and the training loop all share one fundamental flaw: **time is external**.

- VDT's wave recurrence index `t` is an integer step counter with no thermodynamic meaning
- SDM's `cosine_alpha_bar(t, T=200)` is a fixed schedule over 200 preset steps
- Training epochs are external wall-clock steps

This is precisely the **Wheeler-DeWitt problem of time**: a closed system described by a time-independent Hamiltonian has no built-in mechanism to sequence its own evolution. Barontini (2026) solves this experimentally using a Bose-Einstein condensate partitioned into a "dark" (unobserved) and "bright" (observed) sector, and shows that an *entropic time* constructed entirely from internal entropy exchange between the sectors can replace the external laboratory clock — and reproduce the full measured dynamics via an effective Schrödinger equation parametrised by that internal time.

The ESDM applies this lesson directly: every external time parameter in VDT and SDM is replaced by an entropic time derived from spectral entropy exchange between the latent (dark) and active (bright) sectors of the semantic system.

---

## 2. Two-Sector Partition

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

The total semantic Hamiltonian is therefore:

$$\hat{H}_\mathit{sem} = \hat{H}_\mathit{item} + \hat{H}_\mathit{spectral} + \hat{H}_\mathit{wiring}(V)$$

### 2.3 Why the partition matters

- **Item sector** is observable: we can measure $\hat{x}$, compute reconstruction loss, and inspect the active mode population.
- **Spectral sector** is latent: it governs the dynamics of item-space but is never directly observed during inference.
- Entropy flows *between* sectors even though the total system entropy is conserved — exactly as in Barontini's miniuniverse. This is what the `entropy conservation` term in the ELBO enforces (see Section 5).

---

## 3. Entropic Time

### 3.1 Definition

Barontini defines entropic time (Eq. 3 of the paper) as:

$$\tau(\lambda) = \frac{\sigma}{k_B} \int_\lambda \frac{dS}{d\phi} \, |d\phi|$$

This is monotonically increasing by construction: as long as $dS$ and $d\phi$ share sign, the arrow of time cannot reverse. For the ESDM, the clock field $\phi \equiv \log\lambda_S$ is already present in SDM's `log_lam_s`, and the entropy in the bright sector is tracked by `kl_tau`. The discrete entropic time accumulator is:

$$\tau_n = \sum_{i=1}^{n} |\Delta \, \mathit{kl\_tau}_i|$$

where $\Delta \mathit{kl\_tau}_i$ is the per-step change in the tau-mode KL term of the VDT ELBO.

### 3.2 Key properties

1. **Monotonic** — the sum of absolute values cannot decrease, so the arrow of time is guaranteed.
2. **Internally derived** — computed entirely from observable quantities (`kl_tau`) with no external clock.
3. **Stops when entropy is not flowing** — if `kl_tau` is stable, $\tau_n$ stops increasing. This is the semantic equivalent of Barontini's observation that "no entropic time elapses between a big crunch and the subsequent big bang, because no entropy is exchanged there."
4. **Heat death detection** — when $\tau_n$ stalls for many consecutive steps, the system has converged to a stationary state. This is a training termination criterion, not an epoch count.

### 3.3 Non-monotonicity resolution

The clock field `log_lam_s` does not evolve monotonically — it can increase and decrease across training. Barontini explicitly identifies this as the non-monotonicity problem for recollapsing minisuperspace models, and resolves it via the absolute-value integral. The ESDM inherits this fix: the entropic time $\tau$ is always increasing even when `log_lam_s` oscillates.

---

## 4. The Entropic Schrödinger Equation for Semantics

### 4.1 Barontini's derivation

Starting from the static WDW equation and applying a Feshbach-Villars decomposition (keeping only forward-in-time solutions), Barontini obtains (Eq. 6):

$$i\hbar \,\partial_\tau \psi(\tau, a) = \Phi(\tau)\psi(\tau, a) + \Lambda(\tau) \, H_\mathit{geom} \, \psi(\tau, a)$$

where $\Phi(\tau) = \alpha\omega\phi^2 d_\tau\phi$ is a global phase and $\Lambda(\tau) = (\partial_\phi S)^{-1} k_B / \sigma\omega\phi$ is the **entropic pump** — an entropy-dependent energy amplifier. When $\partial_\tau \Lambda = 0$ (no entropy flow), this reduces to an ordinary Schrödinger equation.

### 4.2 ESDM wave recurrence

The VDT wave recurrence (from `VibrationalStateBlock`) is:

$$Q_{t+1} = 2Q_t - Q_{t-1} - dt^2 \, L_f Q_t - \gamma \Delta Q + dt^2 \, B_t$$

In the ESDM this becomes, with $dt$ replaced by $\Delta\tau$ (entropic time step) and the forcing term gated by $\Lambda$:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, L_f Q_t - \gamma \Delta Q + \Lambda(\tau) \cdot \Delta\tau^2 \, B_t$$

where:
- $L_f$ is the feature-space Laplacian (ArrowSpace)
- $B_t$ is the attention-based forcing (self-attention + FFN output from `VibrationalStateBlock`)
- $\Lambda(\tau) = |\partial_\phi \mathit{kl\_tau}|^{-1}$ — the discrete entropic pump
- $\Delta\tau = |\Delta \mathit{kl\_tau}_t|$ — the entropic time step at step $t$

**Interpretation of $\Lambda(\tau)$ as pump:**
- When $\Lambda$ is large (entropy derivative small — modes stable), attention forcing is amplified. The system is in a low-entropy-flow regime and the forcing drives exploration.
- When $\Lambda$ is small (entropy derivative large — rapid mode transitions), attention forcing is suppressed. The system is in a high-entropy-flow regime and the wave recurrence runs close to free vibrational dynamics.
- When $\partial_\tau \Lambda < 0$ (pump shrinking), energy is flowing *into* the bright sector — an expansion phase.
- When $\partial_\tau \Lambda > 0$ (pump growing), energy is being extracted from the bright sector — a contraction phase heading toward "big crunch".

### 4.3 WHT Spectral Filter on the Laplacian

The feature-space Laplacian $L_f \in \mathbb{R}^{F \times F}$ (ArrowSpace GW eigenmaps) must be applied in the wave recurrence as $L_f Q_t$. For high-dimensional feature spaces ($F \gg 1$), this matrix–vector product is expensive and its spectrum may contain components that are irrelevant or harmful to the diffusion trajectory.

The **Walsh–Hadamard Transform (WHT) spectral filter** offers an $O(F \log F)$ alternative that selectively zeroes out Laplacian spectral components outside a learned pass-band, using the Hadamard basis as a computationally cheap proxy for the eigenbasis.

#### 4.3.1 WHT filter definition

Let $H_F \in \{-1, +1\}^{F \times F}$ be the normalised Hadamard matrix:

$$H_F = \frac{1}{\sqrt{F}} W_F$$

where $W_F$ is the standard Walsh–Hadamard matrix of order $F$ (required: $F = 2^m$ for integer $m$, or padded to the nearest power of two). The WHT-filtered Laplacian action on state $Q \in \mathbb{R}^{B \times F}$ is computed as:

$$\widetilde{L_f Q} = H_F^\top \bigl(g \odot (H_F L_f H_F^\top)\bigr) H_F Q$$

where $g \in \mathbb{R}^F$ is a learnable **spectral gate vector** initialised to $\mathbf{1}$ (identity — no filtering) and applied element-wise to the WHT-domain Laplacian rows. In practice, since $L_f$ is fixed after `build_laplacian.py`, the WHT-domain Laplacian $\hat{L}_f = H_F L_f H_F^\top$ is **precomputed once** and cached alongside $L_f$.

The recurrence term becomes:

$$\Delta\tau^2 \, L_f Q_t \quad \longrightarrow \quad \Delta\tau^2 \, H_F^\top (g \odot (\hat{L}_f \cdot (H_F Q_t)^\top)^\top$$

Simplified, letting $\hat{Q}_t = H_F Q_t$ (the WHT of $Q_t$, computed in $O(F \log F)$):

$$\widetilde{L_f Q_t} = H_F^\top \bigl(g \odot (\hat{L}_f \hat{Q}_t^\top)^\top\bigr)$$

#### 4.3.2 Frequency interpretation

The Hadamard basis orders vectors by **sequency** (the number of sign changes), which is the Walsh-domain analogue of frequency. Low-sequency components of $Q_t$ correspond to smooth, globally-coherent vibrational modes; high-sequency components correspond to local, noisy oscillations.

The spectral gate $g$ therefore acts as a **pass-band filter** on the Laplacian action:

| Gate region | Effect on diffusion |
|---|---|
| $g_k \approx 1$ for low sequency | Smooth/global Laplacian modes preserved — slow semantic drift |
| $g_k \approx 0$ for high sequency | Local/noisy modes suppressed — prevents diffusion from chasing noise |
| $g_k$ learned end-to-end | Optimal pass-band co-adapted with entropic time schedule |

This is the **Laplacian spectral filter directive for the diffusion process** identified as a core research driver: the WHT gate selects which frequency bands of the feature-space Laplacian are allowed to direct the diffusion trajectory.

#### 4.3.3 Full wave recurrence with WHT filter

The complete ESDM wave recurrence, incorporating the entropic time gating (§4.2) and the WHT spectral filter, is:

$$\boxed{Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, \widetilde{L_f Q_t} \;-\; \gamma \Delta Q \;+\; \Lambda(\tau) \cdot \Delta\tau^2 \, B_t}$$

where $\widetilde{L_f Q_t} = H_F^\top (g \odot (\hat{L}_f \hat{Q}_t^\top)^\top)$ and $g \in \mathbb{R}^F$ is a learnable parameter of `VibrationalStateBlock`.

#### 4.3.4 Per-mode WHT scheduling

Following the per-mode entropic time $\tau_k$ already used in the diffusion schedule (§5.3), the gate $g$ can be made **time-varying** by conditioning on $\tau_k$:

$$g_k(\tau_k) = \sigma\!\left(w_k - s \cdot \frac{\tau_k}{\tau_{k,\mathit{max}}}\right)$$

where $w_k$ is a learned bias, $s > 0$ is a sharpness parameter, and $\sigma$ is the sigmoid. This implements a soft **frequency annealing**: early in entropic time all frequencies are passed (global exploration); as $\tau_k$ accumulates, high-sequency components are progressively gated out (local noise suppressed), and the diffusion is channelled into smooth semantic basin descent.

---

### 4.4 The Entropic Clock as Principled Collocation Strategy

Physics-informed neural networks (PINNs) are typically analysed as three separate design choices stacked together: architecture (what the model can represent), loss function (what the model is pushed toward), and **collocation strategy** (where in the domain the physics residual is evaluated, and therefore where the model is actually held accountable). The collocation choice is the least-discussed of the three but arguably the most consequential for convergence.

The ESDM differs from standard PINNs in a critical respect: the physics is not an external residual penalty — it *is* the forward pass. The wave recurrence is executed, not approximated. Nevertheless, the collocation framing is directly applicable, and the entropic clock provides a principled answer to it.

#### 4.4.1 Temporal collocation: provided by the clock

The entropic clock implements **collocation by temporal gating** through three coupled mechanisms:

1. **Temporal silencing.** When $\Delta\tau \approx 0$, the recurrence term $\Delta\tau^2\,\widetilde{L_f Q_t}$ vanishes algebraically. The wave equation is not evaluated — it is silenced. The model is not held accountable to physics in that region of $\tau$. This is collocation by exclusion: the clock tells you *where not* to evaluate the physics residual.

2. **Amplitude modulation via $\Lambda(\tau)$.** The entropic pump scales the stochastic forcing term $B_t$. High $\Lambda$ means the injection is large relative to the wave restoring force — the system is in a high-entropy, high-residual regime. Low $\Lambda$ (or capped) means the forcing is frozen. The pump is therefore a proxy for residual magnitude, and the clock tracks it explicitly. This is collocation by signal: the clock identifies high-residual regions of $\tau$-space by the magnitude of entropy flow.

3. **Schedule-coupled noise via $\bar{\alpha}(\tau)$.** The entropic diffusion schedule $\bar{\alpha}(\tau)$ directly controls the noise level seen by the decoder. When $\bar{\alpha} \approx 1$ (early $\tau$, entropy flowing), the decoder operates near clean data — low residual, easy learning. When $\bar{\alpha} \approx 0$ (late $\tau$, entropy exhausted), the decoder operates near pure noise — maximum residual. Crucially, from exp2/3407: with `kl_decay=0.08`, $\bar{\alpha}(\tau) \to 3.75 \times 10^{-33}$ by step 100 of 400, meaning the collocation was **forced entirely into the trivial region** (pure noise, flat solution) for 75% of the run. The model was not held accountable anywhere interesting.

Together, these three mechanisms mean the entropic clock is **necessary and sufficient for temporal collocation**: it solves the where-in-$\tau$ problem automatically. Explicit collocation on the $\tau$ axis would be redundant.

The `kl_decay` sweep in Exp 3 is therefore not merely a hyperparameter search — it is a **collocation strategy experiment**: it tests how spreading the entropic time budget across the training trajectory changes where the wave recurrence is meaningfully evaluated. Reducing `kl_decay` from `0.08` to `{0.01, 0.03}` is analogous to switching from uniform-time collocation to a schedule that concentrates evaluation in the non-trivial part of the solution manifold.

#### 4.4.2 Spectral collocation: the open gap

What the clock does *not* provide is **spectral collocation** — a mechanism to concentrate the wave recurrence evaluation on the eigenmodes where the residual is currently largest. The WHT gate $g_k(\tau_k)$ does time-varying mode selection, but it anneals on a fixed sigmoid schedule, not adaptively toward high-residual modes.

A residual-driven formulation would define the per-mode residual as:

$$r_k(\tau) = \left\| \hat{Q}_{t,k} - \hat{Q}_{t-1,k} \right\|_2$$

and drive the gate by:

$$g_k \propto \sigma(r_k(\tau) - \theta_k)$$

If mode $k$ is currently oscillating rapidly (large $r_k$), its gate should be open; if it has damped out (small $r_k$), its gate should close — independently of the global clock. This would make $g_k$ an **adaptive spectral collocator** rather than a scheduled annealer, concentrating the Laplacian filter on the modes with the highest current residual.

The correlation between $r_k(\tau)$ and $\tau_k$ (per-mode entropic time) determines how much additional gain a residual-driven gate provides over the current sigmoid annealing. Logging $r_k$ alongside $g_k$ in Exp 3 is the minimal diagnostic needed to answer this (see Open Question Q8).

---

## 5. The Entropic Diffusion Schedule

### 5.1 Problem with the current schedule

SDM uses `cosine_alpha_bar(t, T=200)` — a fixed external schedule over 200 preset steps regardless of what the spectral system is doing. This means:
- Steps where no spectral change is occurring get the same diffusion weight as steps where a major mode transition is happening.
- The schedule cannot adapt to different corpus geometries or different `lam_s` configurations.

### 5.2 Entropic schedule

Replace the fixed schedule with one indexed by accumulated entropic time:

$$\bar{\alpha}(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau}{\tau_\mathit{max}}\right)$$

where $\tau$ is the accumulated entropic time and $\tau_\mathit{max}$ is determined from the total entropy budget of the training run (estimated as $\tau_\mathit{max} = \sum_\mathit{all steps} |\Delta \mathit{kl\_tau}|$, updated as a running estimate).

**Effect:** Steps where `kl_tau` is changing rapidly (large $|\Delta \mathit{kl\_tau}|$) contribute more to $\tau$ and therefore advance the diffusion schedule faster — dissolving semantic structure at precisely the moments when the spectral geometry is actively reorganising. Steps where `kl_tau` is stable contribute negligibly — the diffusion schedule slows down at plateaus automatically.

### 5.3 Per-mode schedule

Following SDM's existing `per_mode_variance`, the entropic schedule can be applied per-mode:

$$\bar{\alpha}_k(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau_k}{\tau_{k,\mathit{max}}}\right)$$

where $\tau_k = \sum |\Delta \mathit{kl\_tau}_k|$ accumulates separately for each mode $k$. Modes that transition early (high $\tau_k$) reach their noise floor early; modes that stay stable longer maintain their signal longer.

---

## 6. Architecture

### 6.1 Module overview

```
Input x  (B, D)
     │
     ▼
┌─────────────────────┐
│  EntropicClock      │  NEW — accumulates τ from |Δkl_tau|
│                     │        computes Λ(τ), emits dark_sector_entropy
└─────────┬───────────┘
          │  τ, Λ(τ), Δτ
          ▼
┌─────────────────────┐
│ VibrationalState    │  VDT — wave recurrence, NOW gated by Λ(τ)
│ Block               │        dt → Δτ, B_t scaled by Λ(τ)
│                     │        L_f Q_t → WHT-filtered L̃_f Q_t  [NEW]
└─────────┬───────────┘
          │  Q_t, ρ_+, ρ_-
          ▼
┌─────────────────────┐
│ SpectralLoading     │  VDT — unchanged
│ Decoder             │        z, U_q → W, ω, S
└─────────┬───────────┘
          │  L(z), N_active, log_lam_s
          ▼
┌─────────────────────┐
│ EntropicDiffusion   │  SDM — cosine_alpha_bar replaced by ᾱ(τ)
│ Decoder             │        per-mode schedule indexed by τ_k
└─────────┬───────────┘
          │  x̂
          ▼
   Prediction / Generation
```

### 6.2 Module specifications

**`EntropicClock`** (new module)
- Inputs: `kl_tau` per step (scalar or per-mode vector)
- Outputs: `tau` (accumulated entropic time), `Lambda_tau` (entropic pump), `delta_tau` (current step size), `dark_sector_entropy` (KL of inactive modes)
- State: running sum $\tau_n$, running estimate of $\tau_\mathit{max}$
- Key method: `step(kl_tau_current, kl_tau_prev)` → updates all outputs

**`VibrationalStateBlock`** (modified from VDT)
- Change: accept `Lambda_tau` and `delta_tau` from `EntropicClock`
- Change: replace `dt` with `delta_tau` in recurrence
- Change: scale forcing term `B_t` by `Lambda_tau`
- **Change (new): apply WHT filter to $L_f Q_t$** — precomputed $\hat{L}_f = H_F L_f H_F^\top$ cached at init; learnable gate $g \in \mathbb{R}^F$ replaces bare Laplacian action; optional time-varying gate $g_k(\tau_k)$ from `EntropicClock.per_mode_tau`
- Everything else unchanged

**`SpectralLoadingDecoder`** (unchanged from VDT)
- Produces `log_lam_s` (clock field $\phi$), `N_active`, `L(z)`
- Feeds `log_lam_s` back to `EntropicClock` as $\phi$

**`EntropicDiffusionDecoder`** (modified from SDM)
- Change: replace `cosine_alpha_bar(t, T)` with `alpha_bar_entropic(tau, tau_max)`
- Change: per-mode schedule uses per-mode `tau_k` from `EntropicClock`
- Everything else unchanged (forward step, LGMRF prior, reverse chain)

### 6.3 Signed interference as confidence

The `SignedDensityMatrix` from VDT tracks $\rho = \rho_+ - \rho_-$. From the quantum-vibrational paper, negative eigenvalues of the amplitude Laplacian $\Lambda_A(t) = D(t) - a(t)a(t)^\top$ signal active destructive interference. In the ESDM this becomes a **prediction confidence signal**:

$$\mathit{confidence}(\tau) = \frac{\rho_+}{\rho_+ + \rho_- + \epsilon}$$

When $\rho_- > \rho_+$ (more destructive than constructive interference), the model is in a decoherent state — structurally analogous to Barontini's "big crunch" where the entropic clock stops. Predictions made in this regime should be flagged as low-confidence.

---

## 7. The ESDM ELBO

The ESDM extends the VDT three-term ELBO with an entropy conservation penalty:

$$\mathcal{L}_\mathit{ESDM} = \underbrace{\mathbb{E}_q[\log p(x \mid z, W)]}_\mathit{reconstruction} - \underbrace{\mathit{KL}(q(z) \,\|\, \mathcal{N}(0,I))}_\mathit{kl\_z} - \underbrace{\mathit{KL}(q(S) \,\|\, p(S|I))}_\mathit{kl\_S} - \underbrace{\mathit{KL}(q(\omega) \,\|\, p(\omega|\tau, L))}_\mathit{kl\_tau} - \underbrace{\beta \cdot \left|S_\mathit{bright}(\tau) + S_\mathit{dark}(\tau) - S_0\right|^2}_\mathit{entropy conservation}$$

**New term: entropy conservation**
- $S_\mathit{bright}(\tau) = \mathit{kl\_tau}$ (KL of active modes)
- $S_\mathit{dark}(\tau) = \sum_{k: \omega_k < \theta} \mathit{KL}(\omega_k)$ (KL of inactive modes)
- $S_0 = S_\mathit{bright}(0) + S_\mathit{dark}(0)$ (total entropy at initialisation)
- $\beta$ is a small weight (suggested: $\beta \in [0.001, 0.01]$, or $\beta = 1/S_0$ for a dimensionless form)

This directly mirrors Barontini's experimental verification that "within our errorbars, the entropy of the whole miniuniverse (dark and bright sector) is constant" across the full run.

---

## 8. Training Diagnostics and Health Monitoring

The complete per-batch monitoring loop:

```python
# per batch
kl_bright    = kl_tau                        # active modes entropy
kl_dark      = sum kl(omega_k) for inactive  # dark sector entropy
tau_n       += abs(delta_kl_bright)          # entropic time accumulator (monotone)
Lambda_tau   = 1.0 / (abs(d_kl_bright/d_phi) + eps)  # entropic pump
conserved    = abs(kl_bright + kl_dark - S_0)         # should → 0
rho_sign     = rho_minus / (rho_plus + eps)           # interference ratio → confidence
# WHT filter health
g_mean       = g.mean().item()               # mean gate activation (1.0 = open, 0.0 = closed)
g_entropy    = -(g * g.log() + (1-g) * (1-g+eps).log()).mean()  # gate uncertainty
# Spectral collocation diagnostic (§4.4.2)
r_k          = (Q_hat_t - Q_hat_t_prev).norm(dim=0)  # per-mode WHT residual
r_g_corr     = corrcoef(r_k, g)              # correlation between residual and gate: → 1.0 ideal
```

### 8.1 Health check table

| Metric | Healthy | Warning | Critical |
|---|---|---|---|
| `tau_n` | Monotonically increasing | Flat for > 50 steps | Flat for > 200 steps (heat death) |
| `conserved` | $< 0.05 \cdot S_0$ | $0.05$–$0.2 \cdot S_0$ | $> 0.2 \cdot S_0$ |
| `rho_sign` | Oscillating then stabilising | Monotonically increasing | $> 0.9$ (near-total decoherence) |
| `N_active` | Grows from 0 to $\geq F/2$ | Stalls at $< F/4$ | Collapses after growing (big crunch) |
| `Lambda_tau` | Varies with mode transitions | Explodes ($> 100$) | NaN |
| `g_mean` | $0.4$–$0.8$ (selective) | $< 0.1$ (over-filtered) | $> 0.99$ (WHT filter inactive) |
| `g_entropy` | High early, low late (annealing) | Constant throughout | NaN |
| `r_g_corr` | Increasing toward 1.0 | Flat near 0 | Negative (gate anti-correlated with residual) |

### 8.2 Regime identification

Using Barontini's language, training passes through identifiable phases:

1. **Big bang** ($\tau_n$ starts growing, `N_active` rising): active modes populating, entropy flowing from dark to bright sector
2. **Expansion** ($\Lambda_\tau$ moderate, `conserved` stable): steady semantic organisation
3. **Maximum extension** (`N_active` plateau, $\tau_n$ slowing): diffusion schedule near $\bar{\alpha}(\tau) \approx 0.5$; WHT gate begins annealing high-sequency components
4. **Contraction / big crunch** (`N_active` falling, `rho_sign` rising): modes collapsing back to dark sector
5. **Heat death** ($\tau_n$ stalled, `conserved` near 0): stationary state — training should stop

---

## 9. Barrier Height Sweep (Experimental Protocol)

Barontini varies the barrier height $V = H_\mathit{coupling}/H^\mathit{max}_\mathit{coupling}$ and observes that low $V$ produces cyclic big bang/big crunch dynamics while high $V$ drives the system toward heat death.

The ESDM analogue is a sweep over the `kl_S` weight (the semantic coupling strength):

| `kl_S` weight | Predicted behaviour | Analogue |
|---|---|---|
| Very low | Many mode activation/collapse cycles, fast $\tau_n$ | Low barrier — cyclic BEC |
| Moderate | Single clean expansion + gradual convergence | Intermediate barrier |
| High | `N_active` barely grows, $\tau_n$ nearly constant | High barrier — heat death |

This sweep is the primary validation experiment for the ESDM. Each value of `kl_S` produces a $\tau_n$ vs. training-step curve directly analogous to Barontini's Fig. 2.

---

## 10. Connection to Thermal Time and ArrowSpace

### 10.1 Thermal time hypothesis

Barontini notes that his entropic time "bears conceptual similarities to the thermal time hypothesis" (Connes & Rovelli, 1994), but differs in being operationally constructed from measurable entropy exchange rather than from the algebraic structure of observables. The ESDM inherits this operational character: $\tau_n$ is computed from `kl_tau` which is a directly measurable quantity in the ELBO.

### 10.2 ArrowSpace dual-space hypothesis

The ArrowSpace hypothesis holds that every vector space is simultaneously a geometric item-space and a semantic feature-space. In the ESDM partition:
- **Item-space** = bright sector (observed, reconstructed, measured via $\hat{x}$)
- **Feature-space** = dark sector (Laplacian eigenpairs, spectral loading, ArrowSpace wiring $L_F$)

The Fiedler vector of $L_F$ plays the role of Barontini's scale factor $a$ — it measures the "size of the semantic universe" in feature-space. The Spectral Complementarity Conjecture from the quantum-vibrational paper (that the ArrowSpace Fiedler vector and the amplitude Laplacian bottom eigenvector tend to $90°$ as graph size grows) maps cleanly onto the dark/bright orthogonality: when one sector is maximally uncertain, the other is maximally determined.

The WHT filter (§4.3) acts directly in the dark sector: it selects which feature-space Laplacian frequencies propagate into the bright sector wave recurrence. The learned gate $g$ therefore encodes the **spectral filter directive** — the learnable boundary between dark and bright sector information flow.

### 10.3 Optimisation as recursive semantic search

The ESDM optimisation is not gradient descent on a fixed landscape but a recursive process: each update changes $L_F$, which changes the dark sector entropy, which changes $\tau_n$, which changes the diffusion schedule *and the WHT gate pass-band*, which changes the reconstruction loss, which loops back to update $L_F$. This is the **optimisation-as-recursive-semantic-search** interpretation — the system is searching its own spectral geometry for a configuration where the two-sector entropy conservation law is satisfied.

---

## 11. Open Questions

1. **Feshbach-Villars decomposition in the discrete case.** Does the analogous decomposition in the VDT recurrence correspond to the `kl_tau` positive/negative components of the mode frequency distribution?

2. **Multiple clock fields.** Both `log_lam_s` and the Fiedler eigenvalue of $L_F$ are candidate clocks. Do they produce the same $\tau_n$ trajectory? Do they diverge near a big crunch?

3. **Quantum bounce vs. singularity.** Can the ESDM be engineered to "bounce" — reactivating modes rather than collapsing — by changing the sign of $\Lambda$ near a crunch?

4. **Loschmidt echo as reversibility test.** Running $\tau$ backwards and measuring recovery of the original embedding distribution is a direct test of the arrow of time in semantic space.

5. **$\beta$ calibration for the conservation term.** Natural choice: $\beta = 1/S_0$ (normalised by initial total entropy), making the conservation penalty dimensionless.

6. **WHT gate initialisation strategy.** Initialising $g$ from the eigenvalue spectrum of $\hat{L}_f$ (gate open only on low-energy bands) may provide a warm start that avoids the "gate-all-closed" collapse during early training.

7. **Relationship between WHT sequency and Laplacian eigenorder.** For sparse feature graphs, WHT sequency and $L_F$ eigenorder are correlated but not identical. Quantifying this gap determines how much the WHT proxy loses relative to exact spectral filtering.

8. **Residual-driven spectral collocation.** The current WHT gate $g_k(\tau_k)$ anneals on a fixed sigmoid of per-mode entropic time. An adaptive alternative drives the gate by the per-mode WHT residual $r_k(\tau) = \|\hat{Q}_{t,k} - \hat{Q}_{t-1,k}\|_2$: open the gate where the residual is large, close it where the mode has damped. The correlation $\mathrm{corrcoef}(r_k, g_k)$ logged in Exp 3 measures how much gain a residual-driven gate would provide over the fixed schedule. If this correlation is low ($< 0.3$) throughout training, the fixed sigmoid is a poor proxy for mode activity and adaptive spectral collocation should be prioritised in Exp 4.

---

## 12. Implementation Checklist

- [x] `EntropicClock` module: accumulates $\tau_n$, computes $\Lambda(\tau)$, emits `dark_sector_entropy`
- [x] Modify `VibrationalStateBlock.forward()`: accept `Lambda_tau`, `delta_tau`; scale `B_t`; replace `dt`
- [x] Precompute WHT-domain Laplacian $\hat{L}_f = H_F L_f H_F^\top$ in `build_laplacian.py`; cache alongside $L_f$
- [ ] Add learnable spectral gate $g \in \mathbb{R}^F$ to `VibrationalStateBlock`; wire WHT filter into recurrence
- [ ] Add optional time-varying gate $g_k(\tau_k)$ with sigmoid annealing (§4.3.4)
- [ ] Add `g_mean` and `g_entropy` to step-level CSV log and `health_check()`
- [ ] Add `dark_sector_entropy` metric to step-level CSV log
- [x] Implement entropic-time `alpha_bar` in `clock.py` and consume it in `diffusion_decoder.py`
- [x] Add entropy conservation term to ELBO in `esdm/elbo.py`
- [x] Add `rho_sign` confidence output to model forward pass (via `clock.step`)
- [x] Add `conserved` health metric to `ESDM.health_check()`
- [ ] Add `r_k` (per-mode WHT residual) and `r_g_corr` to step-level CSV log (§4.4.2, Q8)
- [ ] Experiment: `kl_S` weight sweep (barrier height $V$ analogue) → produce $\tau_n$ vs. step curves
- [ ] Experiment: verify entropy conservation law across training run
- [ ] Experiment: Loschmidt echo reversibility test
- [ ] Experiment: WHT gate sequency analysis — plot $g_k$ vs. eigenvalue rank of $\hat{L}_f$ at convergence
- [ ] Experiment: residual-driven gate correlation — plot $\mathrm{corrcoef}(r_k, g_k)$ vs. $\tau$ for Exp 3 runs

---

## References

- Barontini, G. (2026). *Testing the problem of time with cold atoms*. Phys. Rev. Research 8, L022047. DOI: 10.1103/1h9j-df4k
- Connes, A. & Rovelli, C. (1994). Von Neumann algebra automorphisms and time-thermodynamics relation. Class. Quantum Grav. 11, 2899.
- Page, D.N. & Wootters, W.K. (1983). Evolution without evolution. Phys. Rev. D 27, 2885.
- Kiefer, C. & Sandhöfer, B. (2022). Quantum cosmology. Z. Naturforsch. A 77, 543.
- Moriondo, L. (2025). *Vibrational Deductive Transformer*. github.com/tuned-org-uk/vibrational-deduction-transformer
- Moriondo, L. (2025). *Real Amplitudes and Signed Interference in Vibrational Systems*. github.com/tuned-org-uk/quantum-vibrational
- Moriondo, L. (2025). *Spectral Diffusion Model*. github.com/tuned-org-uk/spectral-diffusion
- Moriondo, L. (2024). *ArrowSpace: Semantic Basins*. tuned.org.uk/posts/020_arrowspace_semantic_basins_part2
