# Issue #22 — Full training on real embeddings

## Todos

### Scaffolding (Phase 1)

1. [x] Create `esdm/experiments/exp4.py` — MNIST PCA-16 training experiment.
2. [x] Create `esdm/experiments/exp5.py` — CIFAR-10 DINO-256 training experiment.
3. [x] Create `esdm/eval/latent.py` — latent-space evaluation harness.
4. [x] Create `tests/test_exp4.py` — fast smoke tests for Exp 4.
5. [x] Create `tests/test_exp5.py` — fast smoke tests for Exp 5 (slow tests optional).
6. [~] `esdm/cli/train.py` unchanged; experiment-specific hyperparameters exposed directly in `exp4.py`/`exp5.py`.
7. [ ] Wire `scripts/verify_fitness.py` to accept `--dataset {mnist,cifar10}`.

### Training loop (Phase 2)

8. [x] Run first real-data training run on MNIST and verify loss decreases.
9. [ ] Run first real-data training run on CIFAR-10 subset and verify loss decreases.
10. [x] Implement latent evaluation metrics: reconstruction MSE, conservation drift, k-NN accuracy.
11. [x] Establish real-data fitness baseline (iter 0: -4.1377).
12. [x] Iterate up to 25 times to improve real-data fitness (in progress: iter 1 kept at -3.8699).

### Future architecture step (deferred)

13. [ ] Wire a real feature-space Laplacian $L_F$ into `ESDM` instead of the identity `L_base`.
        - Build empirical $L_F$ from each data batch.
        - Feed it into the vibrational block and spectral decoder.
        - Expected effect: the wave recurrence uses real semantic geometry instead of a placeholder.

### Ship

14. [x] Run full test suite and lint.
15. [x] Commit with `experiment:` prefix referencing #22.
16. [x] Push to origin/main.
