## KL regularisation variants

After the base experiments are convincing, let's start to do some parameters sweeping for weights assigned to the regularisation terms.

[Likely] If you just scale all KLs with one global factor, you’ll keep the math tidy and miss most of the control you actually want.

Let’s carve out useful weighting variants **specific to ESDM** and what each KL term is doing.

## 0. Baseline: current ELBO structure

You effectively have:

\[
\mathcal{L}_\text{ESDM}
= \underbrace{\mathbb{E}_q[\log p(x\mid z,W)]}_{\text{reconstruction}}
- \beta_z \,\text{KL}_z
- \beta_S \,\text{KL}_S
- \beta_\tau \,\text{KL}_\tau
- \beta_\text{cons} \,\text{cons\_entropy}
\]

with the conservation term enforcing \(S_\text{bright}(\tau) + S_\text{dark}(\tau) \approx S_0\).

The question is how to choose \(\beta_z, \beta_S, \beta_\tau, \beta_\text{cons}\) in ways that respect the entropic clock and ArrowSpace geometry, instead of arbitrary β‑VAE style tweaking. [emergentmind](https://www.emergentmind.com/topics/elbo-optimization-schemes)

## 1. Clock‑centric weighting (protect KL_tau)

[Certain] KL_tau is the *clock fuel*; its weight must not be annealed away.

Variant A: **freeze \(\beta_\tau = 1\), scale others relative to it**

- \(\beta_\tau = 1\) always. KL_tau is the reference scale.
- \(\beta_z \in [0, 1]\): start near 0 for synthetic experiments (1–3), ramp to ~0.5–1 in MNIST/CIFAR runs.
- \(\beta_S \in [10^{-3}, 10^{-1}]\): treat KL_S as a *barrier height* knob; sweep this in Exp 3 as you already do.
- \(\beta_\text{cons} \approx 1/S_0\): normalise conservation by initial total entropy as suggested in the design doc.

What this does:

- Keeps the entropic time \(\tau_n = \sum |\Delta \text{KL}_\tau|\) numerically meaningful.
- Lets you test whether strong KL_S or strong conservation is actually needed: if the clock behaviour (heat death timing, \(\tau_n\) shape) is stable as you dial \(\beta_S, \beta_\text{cons}\), you know those are “nice to have”, not critical.

Use this for:

- Exp 1–3: \(\beta_z \approx 0\), \(\beta_\tau=1\), sweep \(\beta_S\) and \(\beta_\text{cons}\) as part of the barrier experiment.
- Exp 4–5: raise \(\beta_z\) once you care about generalisation.

## 2. Barrier‑aware weighting (link KL_S to KL_tau)

[Likely] For the barrier sweep, you don’t want KL_S to be a static knob; you want it **proportional to KL_tau activity** so that when the bright sector is quiet, you don’t over‑penalise the wiring.

Variant B: **dynamic \(\beta_S(\tau)\) tied to clock signal**

Define a time‑dependent weight:

\[
\beta_S(\tau) = \beta_{S,0} \cdot \left( 1 - \exp\left(-\kappa \,\frac{\tau}{\tau_\text{max}}\right) \right)
\]

- Early in entropic time (\(\tau \approx 0\)): \(\beta_S(\tau) \approx 0\) → wiring Laplacian free to adapt.
- Late time (\(\tau \to \tau_\text{max}\)): \(\beta_S(\tau) \to \beta_{S,0}\) → barrier term clamps semantics once the clock says “we’re near convergence”.

Why this fits ESDM:

- Barontini’s barrier height only really bites near maximum extension; early phases are dynamic. [pnas](https://www.pnas.org/doi/10.1073/pnas.2522504123)
- You avoid freezing L(z) too early, which is exactly the premature heat‑death / trivial basin risk you’ve already seen when the schedule collapses.

Implementation pattern:

- Keep logging \(\tau_n\) and \(\tau_\text{max}\).
- In ELBO code, multiply KL_S by \(\beta_S(\tau)\) rather than a fixed scalar.
- Sweep \(\beta_{S,0}\) and \(\kappa\) in Exp 3 instead of a static KL_S weight.

## 3. Conservation vs. expressivity trade‑off (β_cons schedule)

[Likely] The conservation term can be too rigid early on, blocking the model from exploring useful non‑conserved transients; Barontini’s “within errorbars” line is your escape hatch.

Variant C: **late‑onset conservation**

Use a piecewise weight:

\[
\beta_\text{cons}(\tau) =
\begin{cases}
0 & \text{if } \tau / \tau_\text{max} < \eta \\
\beta_0 / S_0 & \text{if } \tau / \tau_\text{max} \ge \eta
\end{cases}
\]

- Pick \(\eta \in [0.3, 0.5]\): conservation only kicks in after the first expansion phase.
- \(\beta_0\) ~ 1 gives you the “within errorbars” regime.

Effect:

- Early training: dark and bright sectors can temporarily violate the conservation law to find a good mutual configuration.
- Late training: conservation starts enforcing the “miniuniverse entropy constant” property once you’ve passed max extension. This matches the story in §8 regime identification: Big bang → expansion → max extension → contraction → heat death.

You can see this directly in the `conserved` metric you already log.

## 4. Latent KL_z: β‑ESDM rather than β‑VAE

[Likely] Using a standard β‑VAE schedule (linearly ramp β_z from 0 to >1) will just re‑open the old “posterior collapse” vs. “over‑compressed latent” trade‑off. In your case, KL_z doesn’t interact with the clock directly, so you can be more aggressive. [emergentmind](https://www.emergentmind.com/topics/elbo-optimization-schemes)

Variant D: **task‑phase β_z schedule**

- Synthetic / unit tests (Exp 1–3): \(\beta_z = 0\). Ignore the latent prior; you’re testing physics.
- MNIST (Exp 4): \(\beta_z \in [0.1, 0.5]\) with a short warm‑up from 0 over the first ~1000 steps.
- CIFAR‑10 (Exp 5): \(\beta_z \in [0.5, 1]\), possibly coupled to a reconstruction threshold: raise \(\beta_z\) only after MSE drops below a target.

If you want something principled, borrow the ED‑VAE idea: decompose the KL into mutual information and entropy terms and **only regularise the prior part** once mutual information saturates. In practice that’s overkill; you can get most of the benefit by simply conditioning \(\beta_z\) on a “latent overfitting” proxy such as: [arxiv](https://arxiv.org/html/2407.06797v1)

- ratio of reconstruction MSE to KL_z; or
- mutual information estimate between x and z (from ELBO surgery). [cs.columbia](https://www.cs.columbia.edu/~blei/fogm/2022F/readings/HoffmanJohnson2016.pdf)

## 5. Path‑A‑aware KL_tau weighting: don’t touch it, but log its ratio

You asked about assigning weights, but with KL_tau the most important thing is to **monitor its relative scale** rather than rescale it.

Variant E: **diagnostic ratios, fixed weight**

Keep \(\beta_\tau = 1\), but add these derived quantities to your logs:

- \(R_{S/\tau} = \frac{\beta_S \,\text{KL}_S}{\beta_\tau \,\text{KL}_\tau}\): “barrier vs. clock” energy ratio.
- \(R_{\text{cons}/\tau} = \frac{\beta_\text{cons} \,\text{cons}}{\beta_\tau \,\text{KL}_\tau}\): “conservation vs. clock” ratio.

If you see \(R_{S/\tau} \gg 1\) early in training, KL_S is dominating the dynamics; you’re effectively back to a static SDM where the clock is a side show. That’s your signal to lower \(\beta_S\) or delay it as in Variant B.

## 6. Concrete “variant menu” you can try

Putting it all together, here are three coherent variants that respect the entropic design:

| Variant | \(\beta_\tau\) | \(\beta_S\) | \(\beta_\text{cons}\) | \(\beta_z\) | Use case |
|--------|---------------|------------|-----------------------|------------|---------|
| Clock‑only | 1 | 0 | 0 | 0 | Debugging Exp 1–2 physics only. |
| Thermodynamic sweep | 1 | static grid {0.001…1} | \(1/S_0\) | 0 | Your current Exp 3 baseline. |
| Clock‑aware | 1 | \(\beta_S(\tau)\) from Variant B | \(\beta_\text{cons}(\tau)\) from Variant C | phase‑dependent | More faithful Barontini analogue, less over‑regularised. |

I disagree because treating all KLs symmetrically (one global β) makes sense in a vanilla VAE, but in ESDM you’ve explicitly given KL_tau a physical role (time) that the others don’t have. Here’s what I’d do instead: **freeze KL_tau’s weight, schedule KL_S and conservation with τ, and only bring in KL_z when you’re out of the synthetic sandbox.** The risk in your current “all equal” approach is that you quietly turn the clock into just another regulariser, instead of the main axis of the dynamics. [zinkov](https://www.zinkov.com/posts/2018-11-02-decomposing-the-elbo/)

If you pick one variant to implement first, I’d start with the clock‑aware Exp 3 sweep: keep \(\beta_\tau = 1\), make \(\beta_S(\tau)\) grow with \(\tau/\tau_\text{max}\), and delay conservation until after the first big bang/expansion phase.
