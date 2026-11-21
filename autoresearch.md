# Representation autoresearch

## Frozen protocol: round 1

- Baseline: release `v2.0.0`, commit `c674a5c`; CPU, seed 3407.
- Hypothesis: initialise the trainable posterior-mean map from the existing
  frozen graph chart, rather than a random projection. Change no other
  initial parameters, architecture dimensions, readout bounds or loss terms.
- Primary metric: clean-feature mean squared error from an affine probe of
  the posterior mean. Fit each probe on training cases only; lower is better.
  This is linear recoverability, not full-model reconstruction or generation.
- Data: the published 32-node analytic path generator, rank 8, 512 training
  cases (stream 0). New validation: 512 cases, stream 100. New confirmation:
  2048 cases, stream 101. Streams are NumPy SeedSequence spawn keys under
  seed 3407. The old stream-1 test set is development evidence only.
- Search: 25 chart-initialised training runs, the Cartesian product of
  learning rates `[0.00003, 0.0001, 0.0003, 0.001, 0.003]` and maximum
  updates `[16, 32, 64, 128, 256]`. These are a bounded experimental grid:
  logarithmic rates around the existing default, and two through 32 epochs.
- Fixed training: actual ESDM loss, Adam, configured weight decay 0.00001,
  batch size 64, gradient clipping 1.0. Seeded epoch permutations (stream
  102), identical stochastic streams, stop at heat death or maximum updates.
- Controls (two additional training runs): v2.0.0 random mean initialisation
  at its default rate 0.0003 and 32 updates; and random initialisation at
  the selected candidate's rate/budget. The latter controls training budget.
  Also report untrained chart-initialised codes and fixed-chart reconstruction.
- Select solely by validation mean error, breaking exact ties by fewer
  maximum updates then lower learning rate. Save the selected checkpoint
  before generating confirmation cases. Never retune after confirmation.
- Accept the completed search only if confirmation MSE improves by at least
  1% over **both** trained random controls, and the upper limit of each
  paired 95% case-bootstrap difference interval is below zero (2000 draws,
  stream 103). The 1% threshold is a predeclared practical release threshold,
  not a model constant. Intervals are conditional on one seed and fitted
  models; they do not measure training-seed uncertainty.
- Guards: finite losses, gradients, parameters and buffers; unchanged frozen
  chart; bounded gains; full default tests and lint; checkpoint round trip.
  Report sampled-code recovery, encoder singular values and posterior-mean
  variance, KL and training trace. An affine probe can hide latent rescaling.
- Release policy: one patch per **successful completed search**, after
  confirmation and guards; retain every unsuccessful candidate in JSONL/TSV.
  A provisional validation leader is not a successful release.

Run `bash autoresearch.sh`. Evidence goes to
`autoresearch/loop-261121-representation/`; checkpoints remain local and are
attached to the successful release. The baseline mean parameters are frozen
in `baseline-encoder.json` before changing model initialisation. The v2.0.0
notebooks and paper remain historical evidence reproducible from their tag.

## Interpretation

The smooth-path generator favours low graph frequencies by construction.
Improvement here supports graph-informed initialisation on this distribution,
not universal representation quality or improved image generation. Compare
the winner with the untrained chart to distinguish retaining a good prior
from learning a better representation.
