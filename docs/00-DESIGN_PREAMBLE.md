## The Entropic Semantic Diffusion Model (ESDM)

The proposed model is a **closed two-sector semantic system** whose internal dynamics are parametrised entirely by entropy exchange, not by an external schedule.

### Two-Sector Partition

Following the Barontini partition $$\hat{H} = \hat{H}_\text{bright} + \hat{H}_\text{dark} + \hat{H}_\text{coupling}$$, the semantic system is split as:[^1]

$$\hat{H}_\text{sem} = \hat{H}_\text{item} + \hat{H}_\text{spectral} + \hat{H}_\text{wiring}(V)$$

- **Item sector** (bright): the observed embedding reconstruction $$\hat{x}$$, active modes $$\omega_k > \theta$$ — what the VDT calls the `N_active` population
- **Spectral sector** (dark): the latent wiring $L(z)$, inactive modes, the ArrowSpace prior eigenpairs $$(U_{1:q}, \lambda_{1:q})$$ — structurally equivalent to the unobserved BEC sector
- **Coupling** $$V$$: the spectral loading regulariser `kl_S`, whose weight directly controls entropy exchange between the two sectors — the optical barrier height analogue


### Entropic Time Definition

Adapting Eq. (3) from Barontini, the entropic time is defined over the trajectory of the spectral clock field $$\phi \equiv \log \lambda_S$$ (the log of the learnable spectral scale, already present in SDM's `log_lam_s` ):[^1]

$$\tau_n = \frac{\sigma}{k_B} \sum_{i=1}^{n} \left| \frac{\Delta S_i}{\Delta \phi_i} \right| |\Delta \phi_i| = \sum_{i=1}^{n} |\Delta \text{kl\_tau}_i|$$

where $$\Delta \text{kl\_tau}_i$$ is the per-step change in the tau-mode KL from the VDT ELBO . This is monotonically increasing by construction — the arrow of time cannot reverse, solving the non-monotonicity problem identified in the previous analysis.

### The Entropic Schrödinger Equation for Semantics

Barontini derives his effective equation by Feshbach-Villars decomposition of the static WDW equation, yielding:[^1]

$$i\hbar \partial_\tau \psi = \Phi(\tau)\psi + \Lambda(\tau) H_\text{geom} \psi$$

The ESDM analogue replaces this with the VDT wave recurrence , now gated by the entropic pump $\Lambda$:

$$Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 \, L_f Q_t - \gamma \Delta Q + \Lambda(\tau) \cdot \Delta\tau^2 \, B_t$$

where:

- $$L_f$$ is the feature-space Laplacian from ArrowSpace
- $$B_t$$ is the attention forcing term (self-attention + FFN output)
- $$\Lambda(\tau) = (\partial_\phi S)^{-1} k_B / \sigma\omega\phi$$ — the entropy derivative of the spectral sector, directly analogous to Barontini's pump[^1]
- $$\Delta\tau$$ replaces $$dt$$ — steps are now measured in entropic units, not external integer steps

When $$\Lambda(\tau) \approx 0$$ (no entropy exchange, modes stable), the recurrence reduces to free vibrational dynamics. When $$\Lambda(\tau)$$ is large (rapid mode activation or collapse), the attention forcing is amplified — the model is thermodynamically *forced* to attend harder precisely when the spectral geometry is changing.

### Architecture: Four Modules

```
Input x (B, D)
      |
      v
[1. EntropicClock]          -- tracks tau from |Delta kl_tau|, emits Lambda(tau)
      |
      v
[2. VibrationalStateBlock]  -- existing VDT wave recurrence, NOW gated by Lambda(tau)
      |    (Q_t, rho_+, rho_-)
      v
[3. SpectralLoadingDecoder] -- existing VDT decoder: z, U_q -> W, omega, S
      |    (L(z), N_active)
      v
[4. EntropicDiffusionDecoder] -- SDM reverse chain, NOW with tau-indexed schedule
      |    (x_hat)
      v
  Prediction / Generation
```

| Module | Source | New contribution from Barontini |
| :-- | :-- | :-- |
| `EntropicClock` | **New** | Accumulates $\|\Delta \text{kl\_tau}\|$, computes $\Lambda(\tau)$, emits `dark_sector_entropy` |
| `VibrationalStateBlock` | VDT | $\Lambda(\tau)$ gating of forcing term $B_t$ |
| `SpectralLoadingDecoder` | VDT | Unchanged; `N_active` → bright-sector size proxy |
| `EntropicDiffusionDecoder` | SDM | `cosine_alpha_bar` replaced by $\tau$-indexed schedule |

### The Entropic Diffusion Schedule

The current SDM uses a fixed `cosine_alpha_bar(t, T=200)` . Replace it with:

$\bar{\alpha}(\tau) = \cos^2\!\left(\frac{\pi}{2} \cdot \frac{\tau}{\tau_\text{max}}\right)$

where $\tau$ is accumulated entropic time, not a fixed step count. This means:

- Steps where `kl_tau` is changing rapidly get *more* diffusion time (larger $\Delta\tau$), dissolving features at semantically active moments
- Steps where entropy is flat get *less* time — the schedule slows down at plateaus automatically
- $\tau_\text{max}$ is determined empirically as the total entropic time of the training run, not preset


### The ELBO

The ESDM extends the VDT three-term ELBO  with a conservation term:

$\mathcal{L}_\text{ESDM} = \underbrace{\mathbb{E}_q[\log p(x | z, W)]}_\text{reconstruction} - \underbrace{\text{KL}(q(z) \| \mathcal{N}(0,I))}_\text{kl\_z} - \underbrace{\text{KL}(q(S) \| p(S|I))}_\text{kl\_S} - \underbrace{\text{KL}(q(\omega) \| p(\omega|\tau,L))}_\text{kl\_tau} - \underbrace{\beta \cdot |S_\text{bright}(\tau) + S_\text{dark}(\tau) - S_0|^2}_\text{entropy conservation}$

The last term is new: it penalises violation of the conservation law that Barontini verifies experimentally — total entropy of the closed system must remain constant even as entropy flows between sectors. $S_\text{dark}(\tau)$ is the KL of inactive modes; $S_\text{bright}(\tau)$ is `kl_tau` of active modes; $S_0$ is their sum at initialisation.[^1]

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
```

A healthy training run shows `conserved → 0`, `tau_n` monotonically increasing, and `rho_sign` oscillating (mode transitions) then stabilising (convergence). A pathological run shows `tau_n` stalling — the system has reached its "heat death", and either the coupling strength $V$ (i.e., `kl_S` weight) needs reducing, or the ArrowSpace index $I$ needs reselecting via the ELBO Bayes factor.[^1]


[^1] Testing the problem of time with cold atoms, Barontini, 2026, https://journals.aps.org/prresearch/abstract/10.1103/1h9j-df4k
