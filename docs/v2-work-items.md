# ESDM v2.0.0 work items

| Item | Deliverable / file ownership | Acceptance |
|---|---|---|
| W1: model | `esdm/models/`, `esdm/sampling/`, model and sampler tests | Real ESDM forward and Q-space sampling use the bounded readout; raw amplitudes survive; fixed bounds hold after optimisation. Check zero-time and basis cases. |
| W2: encoding and decoding evidence | `notebooks/`, shared demonstration code and measured JSON | Two CPU-executed notebooks, seeded at 3407. Quantify held-out encoding error, run matched readout comparisons and demonstrate the actual model boundary. |
| W3: paper | `docs/paper/esdm-v2.tex` and PDF | State the integral convention, RH assumption, proof of readout bounds, ablations and measured results; separate synthetic evidence from quality claims. |
| W4: integration | Design, packaging, checkpoint policy, README, final checks | Package reports 2.0.0; reproducible commands, test/lint and notebook validation, migration guidance. |

W1, W2 and W3 can proceed against the [fixed mathematical interface](00-DESIGN.md#v2-reconstruction-contract).
The [architecture decision](adr/0001-bound-reconstruction-gains.md) records why
the unit-sphere projection was removed and what stability guarantee is lost.
W2 executes after W1 exposes the model API; W3 consumes W2's measured JSON;
W4 verifies the combined revision. The historical 2.5-D score experiments and
their separate topology samplers are not the entropic-clock signal readout.

## Reproduction and interpretation

From the project root:

```bash
uv sync --locked --extra dev --group notebooks
uv run --no-sync python notebooks/run.py
uv run --no-sync pytest tests/ -q
uv run --no-sync ruff check esdm/ tests/ notebooks/run.py
```

The [encoding notebook](../notebooks/encoding-quality.ipynb) measures clean
held-out signal recovery at different chart ranks. Its fitted linear probes
measure the information accessible in the posterior. It also compares the
same rank-eight posterior before and after a short end-to-end training run;
the probe is not the model's reconstruction. The [decoding notebook](../notebooks/decoding.ipynb)
compares five readout treatments on noisy training signals and evaluates them on
shared held-out observations. The [receipt](paper/v2-results/experiment.json)
records the configuration, data hashes, before/after errors, paired intervals
and a full-model check. Rerun it when a hashed model or notebook changes.

For a plain PyTorch state dictionary, retain the training configuration
including the chart dimension. The repository checkpoint producer
(`scripts/validate_generation.py`) bundles the v2 model and clock configuration;
the generation CLI restores the trained frame, readout logits and clock.
Previous-model generation requires the earlier code, not a partially loaded v2.

## Acceptance evidence

| Item | Evidence in this revision |
|---|---|
| W1 | Zero-step raw-state and large-amplitude tests, sampler state restoration, readout bounds and strict checkpoint round trip in `tests/`. |
| W2 | Both notebooks executed on CPU; JSON outputs matched the saved receipt. All 12 recorded file hashes verified. |
| W3 | [Revised TeX](paper/esdm-v2.tex) and [PDF](paper/esdm-v2.pdf), using the same held-out measurements. |
| W4 | Full default suite: 137 passed and five opt-in slow tests skipped. Ruff passed. The local wheel was verified against the package sources. |
