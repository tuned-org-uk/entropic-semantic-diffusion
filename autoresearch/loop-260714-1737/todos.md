# Issue #28 — Follow-up: lr tuning, WHT Phase 2, CIFAR-10 baseline

## Todos

1. [x] Find the right `lr` — **lr=0.05** (sweep 1e-5 to 0.1)
2. [x] Run 800 steps — loss 53.6→40.0, tau_n 0→20.4, alpha_bar 1.0→0.0, no heat death
3. [x] Integrate WHT gate $g$ into `VibrationalStateBlock._step()` (Phase 2) — heat-kernel gate, measurable delta vs none
4. [ ] CIFAR-10 baseline with refactored architecture
5. [ ] Wire `verify_fitness.py` to exp4
