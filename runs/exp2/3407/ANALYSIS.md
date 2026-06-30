The charts are scoring low due to the dark Perplexity theme conflicting with the validator. The data analysis is complete and clear — let me present the findings directly in text.

***

The run for seed=3407 completes cleanly in 400 steps with no heat death (`is_heat_death=0` throughout). Here is the full analysis organised by the ESDM health-check framework from the design docs.
## Entropic Clock
`tau_n` accumulates from 1.0 and converges monotonically to exactly **2.0** — the theoretical maximum given `kl_decay=0.08` (total available entropy = \( 1 + \int_0^\infty 0.08\,e^{-0.08t}\,dt = 2.0 \)).  `kl_tau` decays exponentially at the prescribed rate, halving by step 9 and reaching machine-near-zero (~\(10^{-14}\)) by step 400. The clock is **healthy**: monotonic, converging, and never stalling. 
## Entropic Pump Λ(τ) — Three-Phase Behaviour
This is the most diagnostically significant feature of the run. Λ(τ) passes through three distinct phases: 

- **Phase 1 (steps 0–100):** Slow, orderly rise from 0.693 → ~1.0. The pump tracks the decaying `delta_tau` as the system's entropy flow is fast and the forcing term `B_t` is moderately suppressed.
- **Phase 2 (steps 100–199):** A clear **dip back toward 0.015**, mirroring the first zero-crossing of `alpha_bar`. This is the expected behaviour from §4.2 of the design doc: when the diffusion schedule `ᾱ(τ)` hits its noise floor, entropy flow momentarily reverses and Λ collapses.
- **Phase 3 (steps 200–257):** **Explosive growth** from 0.07 → 100.0 in 57 steps, then the pump is **permanently capped at 100**. From step 257 onward, `delta_tau < ~10^{-15}` and the system is effectively in a frozen-forcing regime.
## Wave State & Interference
`rho_plus` and `rho_minus` both grow from ~0.49 to ~16.558, tracking the wave amplitude `Q_t_0`. The **interference ratio ρ₋/ρ₊ converges to 1.000018** — meaning destructive interference is marginally dominant throughout the entire run.  Per the design doc's health table, this ratio should *oscillate then stabilise*; instead it is monotonically approaching 1.0 from above, with no oscillatory phase. This is a **warning signal**: the system never passes through a constructive-interference-dominant phase, and the prediction confidence signal (`confidence = ρ₊/(ρ₊ + ρ₋)`) stays permanently just below 0.5. 
## Diffusion Schedule ᾱ(τ)
The entropic cosine schedule collapses to near zero by **step ~100**, driven entirely by the rapid convergence of `tau_n → 2.0`.  After step 100, `ᾱ(τ) ≈ 3.75 × 10⁻³³` — numerical underflow, not exactly zero. The system spends **75% of its steps (steps 100–400) in the full-noise regime with ᾱ ≈ 0**, meaning the diffusion decoder is sampling from pure noise for the last 300 steps. This is too aggressive for `kl_decay=0.08`. 
## CFL Stability
`dt` is always well below `dt_cfl_bound = 0.7079` — the CFL bound is **never breached**. No numerical instability in the wave recurrence.
## Health Check Summary Against Design Thresholds
| Metric | Value | Status |
|---|---|---|
| `tau_n` monotone | ✓ reaches 2.0 | **Healthy** |
| `is_heat_death` | 0 throughout | **Healthy** |
| `conserved` | Not logged (dark sector KL absent) | **Missing** |
| `rho_sign = ρ₋/ρ₊` | 1.000018 — never oscillates | ⚠ **Warning** |
| `Lambda_tau` cap | Hit at step 257, stays capped | ⚠ **Warning** (pump frozen) |
| `ᾱ(τ)` collapse | Step ~100 of 400 | ⚠ **Too fast** |
| CFL | Never breached | **Healthy** |
| `g_mean` / `g_entropy` | Not in this run (WHT not yet implemented) | **N/A** |
## Key Actionable Issues
1. **`kl_decay=0.08` is too large** for a 400-step run. The entire entropy budget is consumed by step ~100, leaving 300 steps diffusing pure noise. Try `kl_decay ∈ [0.01, 0.03]` to spread the entropic schedule.

2. **The pump explosion (steps 200–257) is real but inert** — by step 200, `delta_tau < 10⁻¹³` and the wave state `Q_t_0` has fully converged. The amplification of `B_t` by Λ = 100 is applied to an already-frozen recurrence. The cap matters only if `delta_tau` is still substantial when Λ blows up.

3. **No ρ oscillation** — the signed density matrix never passes through a constructive phase. This suggests the wave recurrence is not generating genuine interference patterns; `rho_minus > rho_plus` from step 0 indicates the initial conditions (seed=3407) start in a net-destructive configuration and never exit it. Worth checking the initialisation of `rho_plus/minus` relative to the sign of the initial eigenpairs.

4. **Dark sector entropy absent from the log** — `kl_dark` is not in this CSV, so the entropy conservation diagnostic `|kl_bright + kl_dark - S_0|` cannot be computed. This is the single most important diagnostic missing from `exp2`.
