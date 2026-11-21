# Bound reconstruction gains, not vibrational amplitudes

**Status:** Accepted for ESDM 2.0.0.

**Context.** In 0.4.1 each recurrence step projected its wave state onto the
unit sphere. Two states in the same direction could therefore become
indistinguishable even when their lengths differed. Gaussian posterior draws
alone did not make the downstream wave amplitudes free. The research question
is whether controlling how directions contribute to reconstruction is useful.

**Decision.** Preserve the wave's length as a degree of freedom and constrain
the diagonal entries of the signal readout in the frozen graph basis. The
[design](../00-DESIGN.md#v2-reconstruction-contract) specifies trainable
intervals, their prime-count initialisation, and how the orthogonal complement
is transmitted. Apply the same learned operator to a wave whether it arrives
from training or from sampling. The entropic clock still controls the signal
versus noise mixture.

**Consequences.** The gain limit is inspectable and independent of training
data: a model can be tested at a large input amplitude without mistaking
clipping for reconstruction. This trades the old unit-norm stability
constraint for amplitude expressivity. Damping and the time-step rule remain
useful but do not guarantee stability of the learned, forced recurrence.
Narrow gain intervals can also prevent useful denoising. Both trade-offs
require measurement. Previous checkpoints are not behaviourally equivalent.
Keeping the exact chart with the weights permits reproducible inference even
when an eigenvalue has multiple valid eigenvectors.
