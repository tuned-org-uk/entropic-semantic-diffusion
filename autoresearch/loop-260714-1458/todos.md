# Issue #22 — Full training on real embeddings

## Todos

### Scaffolding (Phase 1)

1. [ ] Create `esdm/experiments/exp4.py` — MNIST PCA-16 training experiment.
2. [ ] Create `esdm/experiments/exp5.py` — CIFAR-10 DINO-256 training experiment.
3. [ ] Create `esdm/eval/latent.py` — latent-space evaluation harness.
4. [ ] Create `tests/test_exp4.py` — fast smoke tests for Exp 4.
5. [ ] Create `tests/test_exp5.py` — fast smoke tests for Exp 5 (slow tests optional).
6. [ ] Update `esdm/cli/train.py` if needed for experiment-specific hyperparameters.
7. [ ] Wire `scripts/verify_fitness.py` to accept `--dataset {mnist,cifar10}`.

### Training loop (Phase 2)

8. [ ] Run first real-data training run on MNIST and verify loss decreases.
9. [ ] Run first real-data training run on CIFAR-10 subset and verify loss decreases.
10. [ ] Implement latent evaluation metrics: reconstruction MSE, conservation drift, k-NN accuracy.
11. [ ] Establish real-data fitness baseline.
12. [ ] Iterate up to 25 times to improve real-data fitness.

### Ship

13. [ ] Run full test suite and lint.
14. [ ] Commit with `experiment:` or `feat:` prefix referencing #22.
15. [ ] Push to origin/main (requires explicit approval).
