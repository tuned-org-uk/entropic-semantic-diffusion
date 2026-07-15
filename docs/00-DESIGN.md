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

### 4.4 The Entropic Clock as Principled Collocation Strategy

Physics-informed neural networks (PINNs) are typically analysed as three separate design choices stacked together: architecture (what the model can represent), loss function (what the model is pushed toward), and **collocation strategy** (where in the domain the physics residual is evaluated, and therefore where the model is actually held accountable). The collocation choice is the least-discussed of the three but arguably the most consequential for convergence.

The ESDM differs from standard PINNs in a critical respect: the physics is not an external residual penalty — it *is* the forward pass. The wave recurrence is executed, not approximated. Nevertheless, the collocation framing is directly applicable, and the entropic clock provides a principled answer to it.

#### 4.4.1 Temporal collocation: provided by the clock

The entropic clock implements **collocation by temporal gating** through three coupled mechanisms:

1. **Temporal silencing.** When $\Delta\tau \approx 0$, the recurrence term $\Delta\tau^2\,L_f Q_t$ vanishes algebraically. The wave equation is not evaluated — it is silenced. The model is not held accountable to physics in that region of $\tau$. This is collocation by exclusion: the clock tells you *where not* to evaluate the physics residual.

2. **Amplitude modulation via $\Lambda(\tau)$.** The entropic pump scales the stochastic forcing term $B_t$. High $\Lambda$ means the injection is large relative to the