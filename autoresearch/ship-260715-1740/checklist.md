# Ship Checklist — Issue #20

## Code PR Checklist

- [x] Tests pass — 78 passed, 5 skipped
- [x] Lint clean — ruff check: All checks passed!
- [x] No secrets in diff — config files only contain model hyperparameters
- [x] H operator implemented — Phi (lifting) + Diffuse (Laplacian heat) + Pi (Rayleigh gradient)
- [x] --q CLI flag added — default q = F/4
- [x] net_energy_flow diagnostic added — in fitness.py and health_check
- [x] pi_contraction_ratio diagnostic added — in fitness.py and health_check
- [x] Frozen-vs-learned H ablation run — delta = +4.51 (9.1%)
- [x] Fitness > baseline — 53.91 (was 46.58 pre-H)
- [ ] Fitness > 110 — N/A (real-data fitness max is 85, not 123 as in synthetic)
- [x] Parameter sweep completed — eta=10, tau_max_est=150 is optimal

## Issue #20 Checklist

- [x] VibrationalStateBlock._step() implements full H (lifting Phi + diffusion + projection Pi)
- [x] --q CLI flag added (default q = F/4)
- [x] net_energy_flow added to esdm/eval/fitness.py
- [x] pi_contraction_ratio added to esdm/eval/fitness.py
- [x] Frozen-H ablation run and documented
- [ ] Fitness score > 110 — re-calibrate target for real-data fitness (max 85)
- [x] All tests pass
