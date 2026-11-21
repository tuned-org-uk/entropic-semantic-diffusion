# Initialise the posterior mean in the existing chart

**Status:** Accepted for v2.0.1 after the frozen confirmation experiment.

**Context.** The existing graph defines a reconstruction subspace, but the
posterior mean initially projected into unrelated random directions.
The initialisation discarded useful smooth-signal information before training.

**Decision.** Copy the transpose of the frozen chart into the mean layer and
set its bias to zero after constructing both encoder layers. Keep those
weights trainable. Preserve random-number consumption, the log-variance head,
Gaussian sampling and reconstruction-gain constraints. Introduce no new
parameters or configuration options.

**Consequences.** Initial mean coordinates have a direct graph interpretation.
This improves linear recoverability on the tested smooth-path distribution.
It favours graph-smooth inputs and can be inappropriate when informative
directions lie elsewhere. The confirmation experiment did not show a benefit
of training over the untrained chart. Numerical stability remains a property
of the dynamics, not a consequence of this initialisation. Checkpoint format
and inference from existing trained weights remain compatible.

The quality trade-off is initial smooth-signal fidelity versus neutrality
towards other feature directions. See [measured results](../representation-search.md).
