# Ship Summary — issue #29 baseline handoff

**Timestamp:** 2026-07-16T10:56:00+01:00
**Status:** COMPLETE
**Issue:** #29 (Post-v0.2.0 pathway)
**Phase:** Baseline handoff (pre-Phase 1)

## What was shipped

1. **Baseline reproduction** — ran `verify_fitness.py --dataset cifar10 --n-steps 800 --lr 0.025` and reproduced the AGENTS.md v0.2.0 baseline: **74.90/85 (88.1%)**.
2. **runs/ cleanup** — removed 56 pre-v0.1.0 experiment directories (all gitignored, all predating the v0.1.0 tag at 2026-07-15 18:33:28). Zero tracked files affected.
3. **Handoff artifact** — `handoff.json` written with full fitness breakdown, findings, and next-phase steps.

## Baseline fitness breakdown (max 85)

| Criterion | Score | Max | Status |
|---|---|---|---|
| tau_monotone | 10.00 | 10 | OK |
| tau_range_bonus | 5.00 | 5 | OK |
| alpha_bar_decrease | 10.00 | 10 | OK |
| conservation | 15.00 | 15 | **Tautological** — enforced by construction |
| heat_death | 10.00 | 10 | OK (fires at step 610/800) |
| heat_death_timing | 4.38 | 5 | Tuned (eps_rel=0.002, patience=50) |
| rho_sign_stabilizing | 10.00 | 10 | OK |
| recon_decrease | 6.58 | 10 | Decoder near-collapsed |
| net_energy_flow | 0.00 | 5 | **The real physics problem** — Lambda dominates H |
| pi_contraction_ratio | 3.94 | 5 | Passes [0.01, 0.3] briefly, decays to 0.027 |
| **Total** | **74.90** | **85** | **88.1%** |

## Key diagnostics from the run

- `tau_n` accumulates to 186.89 (monotone) ✓
- `alpha_bar` drops 1.0 → ~0 (cosine schedule completes) ✓
- `Lambda_tau` starts clipped at 100, oscillates 0.17–43.4 — **pump dominates throughout**
- `nef` (net energy flow) stays 36–176, never approaches 0 — confirms #29 Phase 2 gap
- `pcr` starts >1.0 (H too strong early), decays to 0.027 (H inert late) — confirms Lambda/H imbalance

## Verification results

- Baseline score: 74.8954 (matches AGENTS.md: 74.90) ✓
- runs/ directory: cleaned (0 dirs, was 56) ✓
- handoff.json: written ✓

## Next: #29 Phase 1 — Validate generation on real data

1. Train a checkpoint on CIFAR-10 (DINO-256 embeddings, 800 steps, lr=0.025)
2. Save model `state_dict` to `checkpoints/`
3. Run reverse-SDE sampler on the checkpoint
4. Decode samples to images via pixel decoder
5. Visual inspection + FID against CIFAR-10 test set
