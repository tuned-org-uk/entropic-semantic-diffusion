# v2.0.1 representation search

The improvement comes from graph-informed initialisation. Training did not
measurably improve linear recovery beyond the untrained chart on this task.

## Measured results

All errors below are clean-feature MSE on 2048 fresh confirmation cases.
Each affine probe was fitted on 512 training cases only.

| Representation | Mean-code probe MSE | Sampled-code probe MSE |
|---|---:|---:|
| Random baseline, rate 0.0003, 32 updates | 0.093318747 | 0.194468528 |
| Random matched control, rate 0.00003, 16 updates | 0.088884205 | 0.192391402 |
| Selected graph-initialised encoder | **0.045820973** | 0.139594157 |
| Untrained graph-initialised encoder | 0.045821006 | 0.139448286 |

The winner's primary MSE decreased by 50.90% against baseline and 48.45%
against the matched control. Paired 95% case-bootstrap intervals for winner
minus control were respectively [-0.049293681, -0.045755770] and
[-0.044753718, -0.041429555]. Both cleared the predeclared 1% practical
improvement and strictly negative interval requirement.

Against the untrained chart, the difference was -0.0000000328, interval
[-0.0000002396, +0.0000001618]. Direct chart projection with transpose
synthesis, without a fitted affine probe, achieved 0.045057895.
The sampled-code results are descriptive measurements with one draw per case.

## Search and limits

The frozen [protocol](../autoresearch.md) specifies 25 graph-initialised
training runs plus two random-initialisation controls. Five learning rates
and five update budgets were tested on one synthetic smooth-path distribution.
Validation chose rate 0.00003 and 16 updates. The confirmation cases were
generated only after the chosen checkpoint was saved.

All 25 candidates completed with finite state and gradients. The first
candidate remained the validation leader; the other 24 were discarded.
Longer or faster training generally damaged recoverability. The selected
mean map retained rank eight, with singular values from 0.997195 to 0.998187.
This result does not establish real-image quality, generative performance,
training-seed robustness or superiority on signals unrelated to the graph.

The implementation changes only the initial trainable mean weights and bias,
using the existing frozen chart. Existing checkpoint weights override that
initialisation. Gaussian amplitudes and bounded reconstruction gains retain
the v2 contract. The selected rate and update budget are an experimental
recipe, not new global model defaults.

## Evidence and reproduction

- [Every trial and trace](../autoresearch/loop-261121-representation/autoresearch.jsonl)
- [Trial table](../autoresearch/loop-261121-representation/results.tsv)
- [Frozen source and data hashes](../autoresearch/loop-261121-representation/protocol.json)
- [Selected checkpoint hashes](../autoresearch/loop-261121-representation/selection.json)
- [Confirmation measurements](../autoresearch/loop-261121-representation/confirmation.json)
- [Release-runtime checkpoint replay](../autoresearch/loop-261121-representation/runtime-validation.json)
- [Paper addendum](paper/esdm-v2.0.1-addendum.pdf)

The measured source is commit `55e830d17af8a8d78a3ac1a1e7573cfd2292b72a`.
The protocol's source hashes refer to the measured commit and PyTorch 2.12.1.
Before release, dependencies were updated after an OSV audit; the release
lock uses PyTorch 2.15.0. Replaying the three frozen checkpoints reproduced
all three primary MSE values exactly on the release runtime. The only later
search-code change sets TSV line endings to LF.
No training was repeated after viewing confirmation. Reproduction is a replay,
not a new independent success.

```bash
uv sync --locked --extra dev --group notebooks
bash autoresearch.sh --output-dir results/representation-reproduction
```

The command refuses to overwrite a run. The three checkpoint assets are
`winner.pt`, `baseline.pt` and `matched-control.pt` on the v2.0.1 release.
They are synthetic 32-feature research models, not MNIST/CIFAR image models.
Each contains its configuration, strict-load state and training trace.
The `model_version` field identifies the compatible v2.0.0 architecture;
`proposed_release` records the candidate patch.

Before confirmation, all three saved checkpoints were reloaded; every state
tensor and validation error matched exactly. Release checks: 143 tests
passed, five optional slow tests skipped; Ruff passed. The final OSV audit
covered 141 locked packages and reported zero vulnerabilities or adverse statuses.
