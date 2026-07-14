# Ship Summary — entropic-semantic-diffusion

## What was shipped

Transition from completed issue #21 (real data loaders) to issue #22 (full training on real embeddings).

## Commit

`2c6e7e6` — experiment: scaffold issue #22 — real-data training (MNIST/CIFAR-10) and latent eval

## Artifacts

- Updated `autoresearch/loop-260714-1200/handoff.json` — final timestamp/scope from issue #21 loop.
- Updated `AGENTS.md` — marks #21 complete, #22 in progress, and adds issue #22 autoresearch-loop settings.
- New autoresearch loop: `autoresearch/loop-260714-1458/` with handoff, results TSV, and todo list.
- New code for issue #22:
  - `esdm/experiments/exp4.py` — MNIST PCA-16 training experiment.
  - `esdm/experiments/exp5.py` — CIFAR-10 DINO-256 training experiment.
  - `esdm/eval/latent.py` — latent-space evaluation harness.
  - `tests/test_exp4.py`, `tests/test_exp5.py`, `tests/test_latent.py` — fast smoke tests + optional slow tests.

## Verification

- Pre-ship full test suite: PASS (118 passed, 5 skipped).
- Pre-ship lint: PASS.
- Post-ship smoke tests: PASS (8 passed, 2 skipped in 2.03s).
- No secrets in diff.

## Remote merge

Remote main contained issue #27 documentation commits (00-DESIGN_HYBRID.md, fitness/implementation/experiment docs). These were merged before push with no conflicts.

## Status

COMPLETE.
