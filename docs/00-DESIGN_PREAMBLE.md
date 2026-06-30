## The Entropic Semantic Diffusion Model (ESDM)

The proposed model is a **closed two-sector semantic system** whose internal dynamics are parametrised entirely by entropy exchange, not by an external schedule.

### Two-Sector Partition

Following the Barontini partition $$\hat{H} = \hat{H}_\mathit{bright} + \hat{H}_\mathit{dark} + \hat{H}_\mathit{coupling}$$, the semantic system is split as:[^1]

$$\hat{H}_\mathit{sem} = \hat{H}_\mathit{item} + \hat{H}_\mathit{spectral} + \hat{H}_\mathit{wiring}(V)$$

- **Item sector** (bright): the observed embedding reconstruction $$\hat{x}$$, active modes $$\omega_k > \theta$$ — what the VDT calls the `N_active` population
- **Spectral sector** (dark): the latent wiring $L(z)$, inactive modes, the ArrowSpace prior eigenpairs $$(U_{1:q}, \lambda_{1:q})$$ — structurally equivalent to the unobserved BEC sector
- **Coupling** $$V$$: the spectral loading regulariser `kl_S`, whose weight directly controls entropy exchange between the two sectors — the optical barrier height analogue


### Entropic Time Definition

Adapting Eq. (3) from Barontini, the entropic time is defined over the trajectory of the spectral clock field $$\phi \equiv \log \lambda_S$$ (the log of the learnable spectral scale, already present in SDM's `log_lam_s` ):[^1]

$$\tau_n = \frac{\sigma}{k_B} \sum_{i=1}^{n} \left| \frac{\Delta S_i}{\Delta \phi_i} \right| |\Delta \phi_i| = \sum_{i=1}^{n} |\Delta \mathit{kl\_tau}_i|$$

where $$\Delta \mathit{kl\_tau}_i$$ is the per-step change in the tau-mode KL from the VDT ELBO . This is monotonically increasing by construction — the arrow of time cannot reverse, solving the non-monotonicity problem identified in the previous analysis.

### The Entropic Schrödinger Equation for Semantics

Barontini derives his effective equation by Feshbach-Villars decomposition of the static WDW equation, yielding:[^1]

$$i\hbar \partial_\tau \psi = \Phi(\tau)\psi + \Lambda(\tau) H_\mathit{geom} \psi$$

The ESDM analogue replaces this with the VDT wave recurrence , now gated by the entropic pump $\Lambda$ and filtered by the WHT spectral gate $g$:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, \widetilde{L_f Q_t} - \gamma \Delta Q + \Lambda(\tau) \cdot \Delta\tau^2 \, B_t$$

where:

- $$\widetilde{L_f Q_t} = H_F^\top (g \odot (\hat{L}_f \hat{Q}_t^\top)^\top)$$ is the **WHT-filtered Laplacian action** (see below)
- $$\hat{L}_f = H_F L_f H_F^\top$$ is the WHT-domain Laplacian, precomputed once
- $$\hat{Q}_t = H_F Q_t$$ is the Walsh–Hadamard transform of the state, computed in $O(F \log F)$
- $$g \in \mathbb{R}^F$$ is a **learnable spectral gate** initialised to $\mathbf{1}$ — selects which Hadamard-basis (sequency) components of $L_f$ direct the diffusion
- $$B_t$$ is the attention forcing term (self-attention + FFN output)
- $$\Lambda(\tau) = (\partial_\phi S)^{-1} k_B / \sigma\omega\phi$$ — the entropy derivative of the spectral sector, directly analogous to Barontini's pump[^1]
- $$\Delta\tau$$ replaces $$dt$$ — steps are now measured in entropic units, not external integer steps

When $$\Lambda(\tau) \approx 0$$ (no entropy exchange, modes stable), the recurrence reduces to free vibrational dynamics filtered through the WHT gate. When $$\Lambda(\tau)$$ is large (rapid mode activation or collapse), the attention forcing is amplified — the model is thermodynamically *forced* to attend harder precisely when the spectral geometry is changing.

### WHT Spectral Filter

The **Walsh–Hadamard Transform (WHT) filter** is the spectral filter directive for the diffusion process. It operates on the feature-space Laplacian $L_f \in \mathbb{R}^{F \times F}$ (ArrowSpace GW eigenmaps) and selects which frequency bands of the dark sector are allowed to propagate into the bright sector wave recurrence.

Let $H_F = \frac{1}{\sqrt{F}} W_F$ be the normalised Hadamard matrix of order $F = 2^m$. The WHT filter transforms the Laplacian action as:

$$L_f Q_t \;\longrightarrow\; \widetilde{L_f Q_t} = H_F^\top \bigl(g \odot (\hat{L}_f \hat{Q}_t^\top)^\top\bigr), \qquad \hat{L}_f = H_F L_f H_F^\top, \quad \hat{Q}_t = H_F Q_t$$

The gate $g$ orders components by **sequency** (Walsh-domain analogue of frequency). Low-sequency components of $Q_t$ encode smooth, globally-coherent vibrational modes (slow semantic drift); high-sequency components encode local, noisy oscillations. The gate therefore implements a **pass-band filter** on the Laplacian action:

| Gate region | Effect |
|---|---|
| $g_k \approx 1$ for low sequency | Smooth Laplacian modes preserved — global semantic organisation |
| $g_k \approx 0$ for high sequency | Local noise suppressed — diffusion follows semantic basins |
| $g_k$ learned end-to-end | Co-adapted with entropic schedule |

The gate can be made **time-varying** via sigmoid annealing over per-mode entropic time $\tau_k$ (§4.3.4 of the full design doc), implementing progressive frequency narrowing as the model converges.

### Architecture: Four Modules

```
Input x (B, D)
      |
      v
[1. EntropicClock]          -- tracks tau from |Delta kl_tau|, emits Lambda(tau)
      |
      v
[2. VibrationalStateBlock]  -- VDT wave recurrence, gated by Lambda(tau)
      |                         L_f Q_t -> WHT-filtered L̃_f Q_t  [NEW]
      |    (Q_t, rho_+, rho_-)
      v
[3. SpectralLoadingDecoder] -- VDT decoder: z, U_q -> W, omega, S
      |    (L(z), N_active)
      v
[4. EntropicDiffusionDecoder] -- SDM reverse chain, tau-indexed schedule
      |    (x_hat)
      v
  Prediction / Generation
```

| Module | Source | New contribution |
| :-- | :-- | :-- |
| `EntropicClock` | **New** | Accumulates $\|\Delta \mathit{kl\_tau}\|$, computes $\Lambda(\tau)$, emits `dark_sector_entropy` |
| `VibrationalStateBlock` | VDT | $\Lambda(\tau)$ gating of $B_t$ + **WHT spectral filter on $L_f Q_t$** |
| `SpectralLoadingDecoder` | VDT | Unchanged; `N_active` → bright-sector size proxy |
| `EntropicDiffusionDecoder` | SDM | `cosine_alpha_bar` replaced by $\tau$-indexed schedule |

### The Entropic Diffusion Schedule

The current SDM uses a fixed `cosine_alpha_bar(t, T=200)` . Replace it with:

$$\bar{\alpha}(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau}{\tau_\mathit{max}}\right)$$

where $\tau$ is accumulated entropic time, not a fixed step count. This means:

- Steps where `kl_tau` is changing rapidly get *more* diffusion time (larger $\Delta\tau$), dissolving features at semantically active moments
- Steps where entropy is flat get *less* time — the schedule slows down at plateaus automatically
- $\tau_\mathit{max}$ is determined empirically as the total entropic time of the training run, not preset


### The ELBO

The ESDM extends the VDT three-term ELBO  with a conservation term:

$$\mathcal{L}_\mathit{ESDM} = \underbrace{\mathbb{E}_q[\log p(x | z, W)]}_\mathit{reconstruction} - \underbrace{\mathit{KL}(q(z) \| \mathcal{N}(0,I))}_\mathit{kl\_z} - \underbrace{\mathit{KL}(q(S) \| p(S|I))}_\mathit{kl\_S} - \underbrace{\mathit{KL}(q(\omega) \| p(\omega|\tau,L))}_\mathit{kl\_tau} - \underbrace{\beta \cdot |S_\mathit{bright}(\tau) + S_\mathit{dark}(\tau) - S_0|^2}_\mathit{entropy conservation}$$

The last term is new: it penalises violation of the conservation law that Barontini verifies experimentally — total entropy of the closed system must remain constant even as entropy flows between sectors. $S_\mathit{dark}(\tau)$ is the KL of inactive modes; $S_\mathit{bright}(\tau)$ is `kl_tau` of active modes; $S_0$ is their sum at initialisation.[^1]

### Signed Interference as a Prediction Signal

The quantum-vibrational paper establishes that the amplitude Laplacian $\Lambda_A(t)$ has negative eigenvalues precisely when signed interference is active . In the ESDM this becomes a **prediction confidence signal**: when $\rho_- > \rho_+$ in the `SignedDensityMatrix` (more destructive than constructive interference), the model is in a decoherent state and predictions should be flagged as low-confidence. This is structurally equivalent to Barontini's observation that the entropic clock stops near a big crunch  — no time flows, no information is being processed, no reliable prediction should be made.[^1]

### Conservation Law as a Training Diagnostic

The complete monitoring loop during training becomes:

```
per batch:
  kl_bright  = kl_tau  (active modes)
  kl_dark    = sum of kl(omega_k) for N_active < threshold
  tau_n      += |Delta kl_bright|          # entropic time accumulator
  conserved  = |kl_bright + kl_dark - S_0| # should -> 0
  Lambda_tau  = d(kl_bright)/d(phi)        # entropic pump
  rho_sign   = rho_minus / (rho_plus + eps) # interference ratio -> confidence
  g_mean     = g.mean()                    # WHT gate activity (1=open, 0=closed)
  g_entropy  = gate entropy                # annealing progress
```

A healthy training run shows `conserved → 0`, `tau_n` monotonically increasing, `rho_sign` oscillating then stabilising, and `g_mean` decreasing toward a selective pass-band (high-sequency components progressively gated out as training converges). A pathological run shows `tau_n` stalling — the system has reached its "heat death", and either the coupling strength $V$ (i.e., `kl_S` weight) needs reducing, or the ArrowSpace index $I$ needs reselecting via the ELBO Bayes factor.[^1]


[^1] Testing the problem of time with cold atoms, Barontini, 2026, https://journals.aps.org/prresearch/abstract/10.1103/1h9j-df4k
