# Issue #22 — CIFAR-10 autoresearch loop (post-fix)

## Context

The previous CIFAR-10 baseline was broken (clock frozen, decoder collapsed).
Three fixes applied in commit 82a6a17:
1. Removed S_0 normalisation of kl_tau in the clock; tau_max_est=10.0 (dimensionless)
2. lr 0.05 -> 3e-4 + 50-step linear warmup
3. Decoder-collapse guard in compute_fitness_real (recon < 1e-3 -> score 0)

## Iteration 0 baseline (post-fix)

- Command: verify_fitness.py --dataset cifar10 --n-steps 800 --max-train-samples 2000 --pca-fit-samples 1000
- Fitness: 46.58
- tau_n=34.60, alpha_bar~0 (drop=1.0), recon=0.0039, heat_death=0

### Score breakdown

| Criterion | Score | Max | Gap |
|---|---|---|---|
| tau_monotone | 10.00 | 10 | — |
| tau_range_bonus | 5.00 | 5 | — |
| alpha_bar_decrease | 10.00 | 10 | — |
| conservation | 9.14 | 15 | 5.86 (conserved_ratio=0.39) |
| heat_death | 0.00 | 10 | 10.00 (never fires) |
| heat_death_timing | 0.00 | 5 | 5.00 (never fires) |
| rho_sign_stabilizing | 10.00 | 10 | — |
| loss_decrease | 2.45 | 10 | 7.55 (loss drops 24.5%) |
| **Total** | **46.58** | **75** | **28.42** |

### Key gaps

1. **Heat death (15 pts)**: kl_tau always has non-zero deltas, so _flat_steps never
   reaches heat_death_patience=200. Options: increase clock eps, decrease
   patience, or increase kl_decay so kl_tau converges.
2. **Conservation (5.86 pts)**: total entropy drifts 39% from S_0. The kl_S term
   dominates and drifts as the spectral decoder adapts.
3. **Loss decrease (7.55 pts)**: loss only drops 24.5% (52.2->39.4). Higher lr or
   more steps would help, but stability limits apply.

## Todos

1. [x] Fix clock units bug (commit 82a6a17)
2. [x] Fix lr + warmup (commit 82a6a17)
3. [x] Add decoder-collapse guard (commit 82a6a17)
4. [x] Verify post-fix baseline: 46.58
5. [ ] Iteration 1: tune heat-death detection (clock eps / patience)
6. [ ] Iteration 2-5: sweep lr in {1e-4, 3e-4, 5e-4, 1e-3}
7. [ ] Iteration 6-10: sweep n_steps in {400, 600, 800, 1200}
8. [ ] Iteration 11-15: sweep wht_filter {none, basic, bivariate}
9. [ ] Iteration 16-20: sweep a_tau_scale {1.0, 2.62, 5.0, 8.0}
10. [ ] Iteration 21-25: best-combo + conservation tuning (kl_S_weight)
