# Macro-Issue: Proceed to Experiment 4 — Per-Mode Entropic Schedule on MNIST

> **Type:** Macro-issue / milestone guidance  
> **Derived from:** `runs/exp3/all_runs.csv` analysis (475 runs, 950 k rows, 2026-07-03)  
> **Status:** Exp 3 sweep complete; results support moving to Exp 4 with three hard pivots  
> **Owner:** TBD  
> **Target:** Close this issue when `scripts/run_exp4.py` runs end-to-end and ESDM-per-mode beats or matches ESDM-global on MNIST-16D reconstruction MSE.

---

## 1. Executive Summary

The Experiment 3 barrier-height sweep executed cleanly (no NaNs, no blow-ups, deterministic monotonic `tau_n`), but it does **not** validate the Barontini-regime predictions because the entropic clock is driven by a synthetic exponential signal rather than by a learned bright-sector KL. The sweep therefore functions as an integration test of the pipeline, not as evidence for cyclic big-bang/big-crunch dynamics.

**Decision:** Do not gate Exp 4 on fixing Exp 3's physics. Proceed to Exp 4 (MNIST, per-mode schedule) immediately, but carry forward three hard lessons:

1. **Use a real loss-driven `kl_tau` signal** — the synthetic `kl_S_weight * exp(-kl_decay * step)` driver must be replaced by the actual ELBO `kl_tau` term in Exp 4.
2. **Do not treat `is_heat_death()` as a training stopping criterion** on real data; use it only as a diagnostic.
3. **Promote Path A/B decision to Exp 4** — Exp 3's `r_g_corr` is degenerate (mostly 0) because the gate sees constant residuals; the Path A vs Path B question must be answered on MNIST with real residuals.

---

## 2. What Exp 3 Actually Demonstrated

| Claim | Evidence | Verdict |
|---|---|---|
| Integration of clock + vibrational block + spectral split + WHT gate | 475 runs finished without crash | ✅ Pass |
| Conservation identity `conserved = kl_tau + dark_entropy` | Max residual `5.7e-14` | ✅ Pass |
| `dark_entropy > 0` throughout | Min observed `0.20` nats | ✅ Pass |
| `tau_n` monotonic | All runs monotone | ✅ Pass |
| `a_tau` monotone decreasing | All runs monotone | ✅ Pass |
| Heat-death closure invariant `g_residual_mean < 0.05` at heat death | Violated in 150/175 runs; max `0.119` | ❌ Fail |
| Pump cap timing `lambda_cap_step > 20%` (relaxed) | 20/100 primary runs violate; caps hit at step 143–578 | ❌ Marginal |
| Path A correlation `r_g_corr >= 0.3` | 0% of runs meet threshold | ❌ Fail |
| Distinct Barontini regimes by `kl_S_weight` | `tau_n` is deterministic scaled integral of synthetic signal | ❌ Inconclusive |
| Cyclic `tau_n` oscillations at low barrier | `tau_n` is strictly monotonic by construction | ❌ Fail |

**Root cause:** `esdm/experiments/exp3.py:351` drives the clock with

```python
kl_tau_signal = kl_S_weight * math.exp(-kl_decay * step)
```

This is independent of the vibrational state. Because `tau_n = sum(|Delta kl_tau|)` and `kl_tau_signal` is a positive decaying exponential, `tau_n` is forced to be monotonic and scales almost linearly with `kl_S_weight`. The resulting "heat death" is just the exponential crossing the `clock_eps` threshold, not a semantic mode-collapse.

### Key numbers from the sweep

- **Runs:** 475 (5 `kl_S_weight` × 3 `kl_decay` × 3 `rho` × 2 `lambda_clip` × 2.1 effective `a_tau_scale` × 5 seeds).
- **Final `tau_n` by `kl_S_weight`:** `0.002 / 0.020 / 0.100 / 0.400 / 2.000` — deterministic, no seed variance.
- **First heat-death step** is fully determined by `kl_S_weight` and `kl_decay`:
  - `kl_decay=0.03`: HD at step 467–697 depending on `kl_S_weight`.
  - `kl_decay=0.08`: HD at step 312–399.
- **`r_g_corr`:** mean `-0.0004`, std `0.056`; non-zero only where `r_k` variance is transiently non-zero early in training.
- **`Q_t_0` zero crossings:** 0 in the primary condition — the state does not oscillate.
- **`rho_sign < 1`:** 88% of all steps, but this is an artifact of initial conditions, not a phase transition.

---

## 3. Guidance for Experiment 4

### 3.1 Scope

Implement Experiment 4 as specified in `docs/02-EXPERIMENTS.md` §Experiment 4, with the following mandatory deviations from the original spec:

| Original spec | Required change | Rationale |
|---|---|---|
| MNIST projected to 16-D PCA | Keep 16-D; also log 8-D and 32-D ablations | Verify that per-mode schedule is robust to latent dimension |
| `n_steps=4` vibrational steps | Start with `n_steps=2`; increase to 4 only after baseline works | Reduce instability surface |
| Single `kl_S_weight=0.01` | Grid `kl_S_weight in {0.001, 0.01, 0.05}` | Find the MNIST cyclic regime; do not assume Exp 3 transfers |
| `use_per_mode=True/False` ablation | Keep the ablation; add `a_tau_scale in {2.0, 5.0}` | Answer Path A authority question on real data |
| Heat death as stopping criterion | **Never stop training on `is_heat_death()`** | On real data heat death is a diagnostic, not a termination condition |

### 3.2 Deliverables

1. `esdm/experiments/exp4.py` — training loop for MNIST-PCA with real ELBO.
2. `scripts/run_exp4.py` — CLI supporting single-run and `--sweep` modes.
3. Two model variants logged side-by-side:
   - `ESDM-global` (`use_per_mode=False`)
   - `ESDM-per-mode` (`use_per_mode=True`)
4. Analysis notebook or script `scripts/analyze_exp4.py` producing:
   - Test MSE vs step for both variants.
   - Spearman `rho(tau_k, explained_variance_k)` at convergence.
   - 10-NN latent accuracy vs step.
   - `r_g_corr` distribution across training.
5. Update `docs/02-EXPERIMENTS.md` §Experiment 4 status to "In progress".

### 3.3 Implementation Checklist

- [ ] Add MNIST dataset loader in `esdm/experiments/exp4.py` (PCA projection + L2 normalisation).
- [ ] Wire `EntropicClock.step()` to the **real** `kl_tau` output from `ESDM` forward pass, not a synthetic schedule.
- [ ] Ensure `SpectralLoadingDecoder` returns per-mode `tau_k` buffers when `use_per_mode=True`.
- [ ] Implement reconstruction MSE on held-out MNIST test set (10k samples).
- [ ] Implement 10-NN latent accuracy evaluator on `z` space.
- [ ] Log full schema columns from Exp 3 plus `recon_mse`, `test_mse`, `tau_k` vector, `explained_variance` vector, `latent_knn_acc`.
- [ ] Add success-criteria assertions in `esdm/experiments/exp4.py`.
- [ ] Run limited sweep (2 seeds × 2 variants × 3 `kl_S_weight` × 2 `a_tau_scale` = 24 runs, 10k steps each).
- [ ] Produce summary plots and update this issue with results.

### 3.4 Success Criteria (non-negotiable)

| Criterion | Threshold | Why |
|---|---|---|
| ESDM-per-mode test MSE ≤ ESDM-global test MSE | `<=` | Per-mode schedule must not harm reconstruction |
| Spearman `rho(tau_k, explained_variance_k) < -0.3` | `< -0.3` | High-variance modes should stabilise earlier |
| 10-NN latent accuracy ≥ 85% | `>= 0.85` | Latent space must be semantically organised |
| All `tau_k > 0` at convergence | `> 0` | No mode collapse |
| `conserved < 0.05 * S_0` after step 5000 | `< 0.05` | Weak entropy conservation holds |
| No NaN/Inf in any buffer | 0 NaN/Inf | Basic numerical stability |

### 3.5 Failure Modes to Watch

- **Per-mode MSE worse than global:** Add `tau_k.clamp(min=eps)` and check that `_per_mode_tau` is updated before the ELBO `kl_tau` is summed.
- **No negative Spearman correlation:** The clock may be collapsing all modes at the same rate; inspect per-mode `tau_k` traces.
- **10-NN accuracy < 70%:** The wiring Laplacian is not capturing digit-class geometry. Visualise Fiedler vector vs labels.
- **Heat death before step 3000:** Increase `heat_death_patience` to 1000 and add a `tau_n` floor / entropy pulse only in diagnostic mode, never as a stop condition.
- **`r_g_corr` remains near zero:** Path A gate is not receiving meaningful residuals; consider Path B (`spectral_bands`) as specified in the Exp 4 doc.

---

## 4. Open Questions for Exp 4 to Answer

1. Does a real learned `kl_tau` produce cyclic `tau_n` dynamics on MNIST, or is it monotonic?
2. What is the MNIST-equivalent "barrier height" regime that yields the best latent organisation?
3. Does `a_tau_scale=5.0` produce a usable `r_g_corr` signal on real data, or should Path B be prioritised for Exp 5?
4. Does the per-mode entropic schedule materially improve reconstruction, or is it only a diagnostic convenience?

---

## 5. Dependencies

- #6 (Exp 3 dim=64/bright_modes=32 amendment) — code structure reused.
- `docs/02-EXPERIMENTS.md` §Experiment 4 — source spec.
- `docs/00-DESIGN.md` §5.3 (per-mode schedule), §6.3 (signed interference as confidence), §10.2 (ArrowSpace latent organisation).

---

## 6. Suggested First Commit

```text
exp4: scaffold MNIST per-mode schedule experiment

- Add esdm/experiments/exp4.py with real ELBO-driven kl_tau
- Add scripts/run_exp4.py with single-run and --sweep modes
- Log recon_mse, test_mse, latent_knn_acc, per-mode tau_k
- Update docs/02-EXPERIMENTS.md status
```

---

## 7. Data Provenance

Analysis performed on `runs/exp3/all_runs.csv` (950 000 rows × 36 columns, 475 runs). No manual edits were made to the data. All quoted statistics are reproducible with:

```python
import pandas as pd
df = pd.read_csv("runs/exp3/all_runs.csv")
```

---

## 8. Implementation Update (2026-07-03)

The Exp 4 scaffold has been implemented and all smoke tests pass:

- `esdm/model.py` — added `use_per_mode` config support; per-feature KLs drive `clock.step()` when enabled.
- `esdm/experiments/exp4.py` — MNIST-PCA loader, ESDM-global/per-mode training loop, evaluators, success criteria.
- `scripts/run_exp4.py` — single-run and `--limited-sweep` / `--sweep` CLI.
- `tests/test_exp4_smoke.py` — 13 fast structural tests.
- `docs/02-EXPERIMENTS.md` — Exp 4 status updated to "In progress".

```bash
.venv/bin/python -m pytest tests/test_exp4_smoke.py -q
# 13 passed
```

Remaining work: run a limited production sweep (10k steps) and verify the full success criteria (MSE, Spearman correlation, kNN accuracy) on real MNIST.
