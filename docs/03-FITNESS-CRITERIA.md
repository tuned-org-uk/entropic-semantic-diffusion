# Fitness Criteria Design Analysis

Mapping `docs/00-DESIGN.md` §8.1 health check table to `esdm/eval/fitness.py`.

> **Update (2026-07-13, issue #27):** added §Phase-aware ablation scoring and
> §Diversity / quality criteria derived from the Un-0 comparison. These extend
> the existing scalar fitness score with group scores that are structurally
> necessary for validating the string/spring spectral mechanism.

## Problem

The current fitness criteria score the residual-gate correlation (`r_g_corr_positive`,
`r_g_corr_increasing`) across the **entire** run, including post-heat-death steps.  The
design doc §8.1 explicitly classifies `r_k_mean` as:

| State | `r_k_mean` |
|---|---|
| Healthy | Positive; decreasing after heat-death onset |
| Warning | Flat after heat death |
| Critical | Increasing after heat death (ghost modes) |

So residuals are **expected** to be flat (zero variance) after heat death.  Any
correlation computed from a zero-variance signal is undefined or zero.  Scoring the
correlation trend across the full run therefore penalises a system that behaves exactly
as the design intends: strong correlation during the active phase, zero after heat death.

The same applies to `r_g_corr_positive`: the design says the healthy trend is
"Increasing toward 1.0", but this is the trajectory **during the active phase**, not the
final value after the system has stopped evolving.

## Fix

Restrict the `r_g_corr` criteria to the **active phase**: steps before the first heat-death
event (or the full run if heat death never fires).

| Criterion | Old | New |
|---|---|---|
| `r_g_corr_positive` | `rgc_final` (post-heat-death, always 0) | `rgc` mean over active phase |
| `r_g_corr_increasing` | second-half mean > first-half mean (full run) | last-third mean > first-third mean (active phase only) |

This aligns the scoring with the design's phase-aware health table.

## Full criteria mapping (design §8.1 → fitness)

| Design metric | Healthy | Fitness criterion | Current | Status |
|---|---|---|---|---|
| `tau_n` | Monotonically increasing | `tau_monotone` (10) + `tau_range_bonus` (5) | aligned | OK |
| `conserved` | `< 0.05 * S_0` | `conservation` (15): `1 - mean(residual)/S_0` | aligned | OK |
| `rho_sign` | Oscillating then stabilising | `rho_sign_stabilizing` (10): variance decrease | partial - only checks variance, not oscillation | Could add oscillation-before-stabilisation check |
| `N_active` | Grows from 0 to >= F/2 | - | not logged in exp3 | **Gap**: needs exp3 instrumentation |
| `Lambda_tau` | Varies with mode transitions | - | not scored | **Gap**: could add "no explosion" check |
| `g_mean` | 0.4-0.8 (selective) | `g_mean_selective` (10) | aligned | OK |
| `g_entropy` | High early, low late (annealing) | - | logged as alias for `dark_entropy` | **Gap**: could add annealing criterion |
| `r_k_mean` | Positive; decreasing after heat-death onset | - | logged but not scored | **Gap**: could add post-heat-death flatness check |
| `r_k_max` | Decreasing monotonically after convergence | - | logged but not scored | **Gap**: could add runaway-mode sentinel |
| `g_residual_mean` | Decreasing; < 0.05 at heat death | `g_residual_decrease` (10) + `g_residual_final_bonus` (5) | aligned | OK |
| `a_tau` | Monotone decreasing to 0 | `a_tau_monotone` (5) + `a_tau_final_bonus` (3) | aligned | OK |
| `r_g_corr` | Increasing toward 1.0 | `r_g_corr_positive` (10) + `r_g_corr_increasing` (5) | scored over full run | **Fixed**: restrict to active phase |

## Improvement pathways

1. **Active-phase scoping (this change).** Restrict `r_g_corr` criteria to pre-heat-death
   steps.  This is the minimal fix that makes the criteria consistent with the design.

2. **Add `g_entropy` annealing criterion.** The design says gate entropy should be
   "high early, low late".  This can be scored as: `g_entropy[0] > g_entropy[-1]` with a
   magnitude bonus.  Requires logging a true gate entropy (not the `dark_entropy` alias).

3. **Add `r_k_mean` post-heat-death flatness.** The design says residuals should be flat
   after heat death.  Score: `std(r_k_mean[post_hd]) < threshold`.  This rewards a clean
   convergence rather than ghost modes.

4. **Add `N_active` growth criterion.** The design says `N_active` should grow from 0 to
   >= F/2.  Requires exp3 to log `N_active` from the vibrational block.  Currently exp3
   uses a synthetic clock, so `N_active` is NaN.

5. **Add `Lambda_tau` stability.** The design says pump should vary with mode transitions,
   not explode.  Score: `max(Lambda_tau) < lambda_clip` and no NaN.

6. **Revisit criterion weights.** The current weights sum to 123, not 100 as documented.
   Either rescale to 100 or update the documentation to reflect the true maximum.

7. **Phase-aware scoring.** Instead of a single composite score, produce per-phase scores
   (big bang / expansion / maximum extension / contraction / heat death) that map to the
   design's §8.2 regime identification.  This would give a richer diagnostic than a single
   number.

---

## Phase-aware ablation scoring _(new, issue #27)_

The scalar fitness score is necessary for continuous training diagnostics but insufficient
for validating the string/spring spectral mechanism. The following **group scores**
correspond to the validation requirements in `001-IMPLEMENTATION-PLAN.md` §8 and must
be reported alongside the scalar score in any ablation result table.

### A. Spectral gate learning score

Measures whether the WHT gate $g$ is learning a meaningful frequency selection
(pass-band behaviour) vs. remaining flat (reservoir behaviour).

| Sub-criterion | Definition | Pass condition |
|---|---|---|
| `gate_std_final` | $\text{std}(g)$ at end of active phase | $> 0.05$ (gate has differentiated; not all-ones) |
| `gate_entropy_drop` | $H(g_0) - H(g_\text{final})$, Shannon entropy of gate values | $> 0$ (entropy decreasing = annealing happening) |
| `gate_low_seq_mean` | Mean $g_k$ for lowest 25% sequency components | $> 0.7$ (low-sequency / smooth modes preserved) |
| `gate_high_seq_mean` | Mean $g_k$ for highest 25% sequency components | $< 0.4$ (high-sequency / noisy modes suppressed) |

In the frozen-gate (Condition B) runs from Experiment 7 (`002-EXPERIMENTS.md`),
`gate_std_final` will be 0 by construction — this is the correct baseline for
confirming that the learned gate is doing something the frozen gate cannot.

### B. Interference / signed-amplitude score

Measures whether destructive interference ($\rho_- > \rho_+$) is occurring during
meaningful parts of the run vs. being a degenerate state.

| Sub-criterion | Definition | Pass condition |
|---|---|---|
| `rho_sign_active_ratio` | Fraction of active-phase steps where $\rho_- / (\rho_+ + \epsilon) > 0.5$ | $\in [0.1, 0.5]$ (interference occurs but does not dominate) |
| `rho_sign_terminal` | $\rho_- / (\rho_+ + \epsilon)$ at heat death | $< 0.2$ (system stabilises into constructive regime) |
| `signed_vs_unsigned_delta` | Chamfer distance delta between Condition A (signed) and Condition C (unsigned) in Exp 4 | $> 0$ (signed model is better; else flag for review) |

`rho_sign_active_ratio` being $< 0.1$ indicates the model never enters the
destructive-interference regime, which means the signed density matrix may be
architecturally decorative. Being $> 0.5$ indicates persistent decoherence, which
the design doc flags as a prediction-confidence failure (analogous to Barontini's
"big crunch" state).

### C. Conservation score (extended)

Extends the existing scalar `conservation` criterion with a trajectory-level check.

| Sub-criterion | Definition | Pass condition |
|---|---|---|
| `conservation_terminal` | $|S_\text{bright} + S_\text{dark} - S_0|$ at final step | $< 0.05 \cdot S_0$ |
| `conservation_max_active` | $\max |S_\text{bright} + S_\text{dark} - S_0|$ during active phase | $< 0.2 \cdot S_0$ (no large transient violations) |
| `conservation_at_N_steps` | `conservation_terminal` at each step count in the unroll-depth sweep | Reported as a curve, not a single number |

`conservation_at_N_steps` connects to Experiment 8: a well-designed reverse
process should converge conservation regardless of step count, not only at
the default $N$. If conservation at $N=1$ is already near zero, the diffusion
trajectory is not doing real work; if it improves monotonically with $N$,
the trajectory is meaningfully constrained by the physics.

---

## Diversity / quality criteria _(new, issue #27)_

These criteria are reported as part of any generative evaluation (Experiments
7–9 in `002-EXPERIMENTS.md`). They are not part of the scalar fitness score
but are required columns in every ablation result table.

| Metric | Definition | Computed from |
|---|---|---|
| `coverage` | Fraction of reference samples with a generated neighbour within Chamfer threshold $\delta$ | `eval/metrics.py` |
| `precision` | Fraction of generated samples within Chamfer threshold $\delta$ of any reference sample | `eval/metrics.py` |
| `mode_separation` | $\text{std}(R(f_i))$ across generated samples, where $R(f_i) = f_i^\top L_F f_i / f_i^\top f_i$ | `eval/metrics.py`, uses `WireGraph.rayleigh_quotient` |

**Interpretation rules:**

- An improvement in `precision` without `coverage` is not a win for the ESDM
  dual-space hypothesis. The hypothesis predicts improved basin coverage
  (higher `coverage`, higher `mode_separation`), not merely sharper samples.
- A full ESDM result that matches a non-spectral baseline on `coverage` and
  `mode_separation` is evidence that the string/spring spectral dynamics
  are not contributing to diversity, regardless of Chamfer distance.
- `mode_separation` being $\approx 0$ across generated samples signals
  spectral mode collapse: all generated samples are in one Rayleigh-quotient
  regime, suggesting the $L_F$-guided dynamics are not exploring the
  feature-space geometry.

---

## Next steps

- [x] Implement active-phase scoping for `r_g_corr` criteria
- [ ] Add `g_entropy` annealing criterion (requires true gate entropy logging)
- [ ] Add `r_k_mean` post-heat-death flatness check
- [ ] Instrument exp3 to log `N_active` for the growth criterion
- [ ] Rescale criterion weights to sum to 100 or document the true maximum
- [ ] Design per-phase scoring aligned with §8.2
- [ ] Implement `gate_std_final`, `gate_entropy_drop`, `gate_low_seq_mean`,
  `gate_high_seq_mean` in `eval/fitness.py` (Spectral gate learning score, §A)
- [ ] Implement `rho_sign_active_ratio`, `rho_sign_terminal` in `eval/fitness.py`
  (Interference / signed-amplitude score, §B)
- [ ] Implement `conservation_terminal`, `conservation_max_active`,
  `conservation_at_N_steps` in `eval/fitness.py` (Conservation score, §C)
- [ ] Implement `coverage`, `precision`, `mode_separation` in `eval/metrics.py`
  (Diversity / quality criteria)
- [ ] Add Condition C (unsigned spectral gate) to experiment harness
  (`eval/harness.py`) for Experiment 4
