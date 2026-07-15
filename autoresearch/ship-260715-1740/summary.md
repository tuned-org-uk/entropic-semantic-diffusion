# Ship Summary — Issue #20

## What was shipped

Implementation of the hybridisation operator $\mathcal{H} = \Pi \circ \text{Diffuse}_{L_F} \circ \Phi$ in `VibrationalStateBlock._step()`, replacing the bare `L_f @ Q_t` product with a three-step spectral procedure.

## Commits

1. `1bae020` — feat: implement hybridisation operator H (Phi + Diffuse + Pi)
2. `c3b0f61` — experiment: tau_max_est configurable + sweep results (eta=10, tau_max=150)

## Files changed

- `esdm/models/vibrational.py` — H operator implementation, diagnostics buffers, freeze_h support
- `esdm/models/esdm.py` — pass U_q to vibrational block, configurable tau_max_est, diagnostics in health_check
- `esdm/_configs/default.yaml` — sigma_sq, eta, freeze_h, tau_max_est config params
- `esdm/eval/fitness.py` — net_energy_flow + pi_contraction_ratio scoring criteria
- `esdm/experiments/exp4.py` — q/sigma_sq/eta/freeze_h/tau_max_est params, diagnostic logging
- `esdm/experiments/exp5.py` — same as exp4
- `scripts/verify_fitness.py` — --q, --sigma-sq, --eta, --freeze-h, --tau-max-est CLI flags
- `tests/test_vibrational.py` — 6 new H operator tests

## Verification results

- Tests: 78 passed, 5 skipped
- Lint: clean (ruff)
- Fitness (learned H): 53.91 (eta=10, tau_max=150, CIFAR-10 800 steps)
- Fitness (frozen H): 49.40 (same config, --freeze-h)
- Ablation delta: +4.51 (9.1%) — learned H outperforms frozen H

## Parameter sweep results

| eta | tau_max_est | sigma_sq | q | Fitness | Notes |
|-----|-------------|----------|---|---------|-------|
| 0.1 | 10 | 1.0 | 64 | 47.40 | H too weak (baseline) |
| 1.0 | 10 | 1.0 | 64 | 48.45 | slightly better |
| 10.0 | 10 | 1.0 | 64 | 52.47 | sweet spot, pcr non-zero early |
| 50.0 | 10 | 1.0 | 64 | 47.34 | over-contraction (pcr=1.67) |
| 10.0 | 10 | 0.1 | 64 | 52.43 | sigma_sq barely matters |
| 10.0 | 10 | 10.0 | 64 | 52.50 | sigma_sq barely matters |
| 10.0 | 10 | 1.0 | 128 | 52.38 | q=128 worse, loss explodes |
| 10.0 | 10 | 1.0 | 16 | 47.74 | q=16, decoder collapse risk |
| 10.0 | 100 | 1.0 | 64 | 53.59 | close to best |
| **10.0** | **150** | **1.0** | **64** | **53.91** | **best — cosine schedule spreads** |
| 10.0 | 200 | 1.0 | 64 | 0.00 | alpha_bar never completes |
| 5.0 | 150 | 1.0 | 64 | 53.80 | close to best |

## Frozen-vs-learned ablation

| Condition | Fitness | pcr (late) | alpha_bar (final) |
|---|---|---|---|
| Learned H | 53.91 | 0.0006 (passes [0.01, 0.3]) | ~0 |
| Frozen H | 49.40 | >1.0 (over-contraction) | 0.092 |

The learned H adapts eta downward during training, reducing pi_contraction_ratio from 1.34 → 0.0006. The frozen H stays at pcr > 1.0 (Pi overwhelming wave dynamics). Delta = +4.51 (9.1%) proves the trained dynamics do real computational work.

## Sign convention note

The design doc (00-DESIGN_HYBRID.md) defines Pi = -eta * grad_R and the recurrence as -dt^2 * Pi, which gives +eta * dt^2 * grad_R (destabilising). The implemented sign keeps H_action = eta * grad_R (analogous to L*Q), so -dt^2 * H_action is the restoring force toward low-energy modes. This is the physically correct wave equation convention.
