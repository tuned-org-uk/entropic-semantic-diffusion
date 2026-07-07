# Fitness Criteria Design Analysis

Mapping `docs/00-DESIGN.md` §8.1 health check table to `esdm/eval/fitness.py`.

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

## Next steps

- [x] Implement active-phase scoping for `r_g_corr` criteria
- [ ] Add `g_entropy` annealing criterion (requires true gate entropy logging)
- [ ] Add `r_k_mean` post-heat-death flatness check
- [ ] Instrument exp3 to log `N_active` for the growth criterion
- [ ] Rescale criterion weights to sum to 100 or document the true maximum
- [ ] Design per-phase scoring aligned with §8.2
