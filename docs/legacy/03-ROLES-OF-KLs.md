## Quick map: what each KL is doing

In the current ESDM ELBO the terms are:

- \( \text{reconstruction}(x \mid z, W) \)  
- \( \text{KL}_z = \text{KL}(q(z) \| \mathcal{N}(0, I)) \) — latent prior. [cs.cornell](https://www.cs.cornell.edu/courses/cs4782/2025sp/slides/pdf/week9_1_slides.pdf)  
- \( \text{KL}_S \) — semantic / wiring KL (your “barrier height” term).  
- \( \text{KL}_\tau \) — clock-sector KL (bright-mode entropy).  
- **Entropy conservation term** tying bright + dark sector to \(S_0\).

Functionally:

| Term | Role in ESDM | What breaks if you drop it? |
|------|--------------|-----------------------------|
| KL_z | Standard VAE regulariser on z; keeps latent sane. [cs.cornell](https://www.cs.cornell.edu/courses/cs4782/2025sp/slides/pdf/week9_1_slides.pdf) | Latent blows up / overfits; clock still works. |
| KL_S | Controls coupling between items and feature Laplacian; drives barrier sweep. | Two-sector “mini-universe” analogy collapses; barrier-height experiment meaningless. |
| KL_tau | Defines bright-sector entropy; *this* is the clock’s fuel. | Entropic time \(\tau_n\) becomes undefined; core design collapses. |
| Entropy conservation | Ensures \(S_\text{bright} + S_\text{dark} \approx S_0\); matches Barontini’s conserved miniuniverse. | Dark/bright split loses physical meaning; clock still runs but isn’t “two-sector”. |

So:

- [Certain] **KL_tau is non‑negotiable.** Without a bright-sector KL, your entropic time \(\tau_n = \sum |\Delta \text{KL}_\tau|\) doesn’t exist, so the whole ESDM idea disappears.  
- [Likely] **KL_S and the conservation term are conceptually important but experimentally optional.** You can turn them down or off to test behaviour; the clock and diffusion schedule still run, you just lose the Barontini thermodynamics guarantee.  
- [Likely] **KL_z is just “normal VAE hygiene”.** You can relax or anneal it during early experiments if it’s fighting the spectral geometry; you’ll get uglier latents but still have a working entropic diffusion model. [cs.cornell](https://www.cs.cornell.edu/courses/cs4782/2025sp/slides/pdf/week9_1_slides.pdf)

## When each KL really matters (per experiment)

The design file already hints at where each KL is doing real work vs. being “pure doctrine”.

- **Exp 1–2 (clock + wave unit tests):**  
  - Needed: KL_tau (to define \(\tau_n\) and \(\Lambda(\tau)\)).  
  - Can be zeroed: KL_z, KL_S, conservation term — synthetic setups don’t need full thermodynamics.  
- **Exp 3 (barrier sweep):**  
  - KL_tau + KL_S + conservation are the trio that make the Barontini analogy real. If you drop KL_S, “barrier height” is just a scalar weight with no semantic coupling interpretation.  
  - KL_z still optional — you’re in 8D clusters; it’s more about spectral dynamics than latent elegance.  
- **Exp 4–5 (MNIST/CIFAR‑10):**  
  - KL_z becomes important: you actually care about generalisation and latent structure. [cs.cornell](https://www.cs.cornell.edu/courses/cs4782/2025sp/slides/pdf/week9_1_slides.pdf)
  - KL_S might be tunable per task: for some datasets you may want weaker coupling to let ArrowSpace discover wiring; for others strong KL_S to force structure.  
  - Conservation term: you can test removing or down-weighting it to see whether strict conservation is helping or just making optimisation brittle.

One concrete ablation path that won’t wreck the model:

1. Early prototyping / debugging: keep only reconstruction + KL_tau; set KL_z, KL_S, conservation weight to ~0.  
2. Once the clock behaviour looks sane, add KL_S and conservation back in for Exp 3.  
3. Only when you care about downstream quality (Exp 4–5) do you anneal KL_z into a proper VAE regime.

## The risk in “keep everything on”

I disagree because if you insist all KL terms are “always necessary” you’re locking yourself into one specific physical narrative: two-sector miniuniverse with perfectly conserved entropy and “correct” latent priors. Here’s what I’d do instead:

- Use **KL_tau as the backbone** — never touch it; it’s defining time.  
- Treat **KL_S and conservation** as knobs controlling “how physical” you want the run to be; dial them down when the optimiser is obviously fighting spectral geometry.  
- Treat **KL_z** as a regulariser you turn up only when you start caring about generalisation, not when you’re still stress-testing spectral collocation.

The downside of your current approach is simple: over-regularisation. In a spectral model this often shows up as premature heat death or trivial basins — exactly what you already saw when the clock schedule collapsed too fast.

If you were to drop one term for a diagnostic run tomorrow, which would you be most comfortable turning off: KL_z, KL_S, or the conservation penalty?
