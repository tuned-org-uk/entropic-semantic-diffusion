# Ship Summary — entropic-semantic-diffusion

## What is being shipped

Transition from completed issue #21 (real data loaders) to issue #22 (full training on real embeddings).

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

- Full test suite: PASS (118 passed, 5 skipped).
- Lint: PASS.
- No secrets in diff.

## Pre-existing unstaged change

`README.md` has an unstaged modification unrelated to this ship (removes historical spelling note). It is NOT included in the proposed commit.

## Status

DRY_RUN complete. Actual git commit/push pending explicit user approval.
