# 001 — ESDM: PyTorch Implementation Plan

> **Status:** Implementation-ready design  
> **Depends on:** `docs/00-DESIGN.md`, `000-DIFFUSION-ON-PHYSICAL-SURFACES.md`  
> **Integrates:** ArrowSpace dual-space fusion analysis, Barontini 2026, SPIN/GW spectral indexing, 2.5-D hypothesis  
> **Target:** PyTorch ≥ 2.3, Python ≥ 3.12

Here's a concise sketch of the update equations followed by a concrete PyTorch implementation and validation plan comparing your "2.5-D" graph-wired diffusion against a full 3D volumetric baseline.

## Update equations

**Baseline (3D volumetric) reverse-diffusion update**, following the standard Fokker–Planck-derived reverse SDE with a learned score $s_\theta$:[^1]

$$
x_{t-\Delta t} = x_t + \left[\frac{\sigma^2}{2}\, s_\theta(x_t,t) \right]\Delta t + \sigma\sqrt{\Delta t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I_3)
$$

This requires $x_t \in \mathbb R^3$ and the score network must implicitly learn the local Jacobian structure of the ambient manifold, i.e. it pays the "volume" cost implicitly through the dimensionality of $s_\theta$.

**Your 2.5-D dual-space update** replaces the ambient Laplacian with the graph Laplacian $L_F$ from graph wiring, and splits the state into a low-dimensional item-coordinate $z_t$ (≤2D) plus a feature-channel signal $f_t$ living on the wired graph:

$$
z_{t-\Delta t} = z_t + \eta\,\nabla_z R(f_t)\,\Delta t + \sigma\sqrt{\Delta t}\,\epsilon_z,\qquad \epsilon_z\sim\mathcal N(0,I_2)
$$

$$
f_{t-\Delta t} = f_t - \frac{\sigma^2\Delta t}{2}\, L_F f_t
$$

where $R(f) = f^\top L_F f / f^\top f$ is the Rayleigh quotient your graph-wiring paper shows converges to Dirichlet energy, and $\nabla_z R$ is the score analogue — a gradient computed purely from sparse Laplacian products, never requiring a determinant. The key structural claim to test is that $(z_t, f_t)$ jointly carries the same generative information as $x_t \in \mathbb R^3$.[^2]

## Implementation plan (PyTorch)

**1. Shared data + graph construction**

- Generate synthetic 3D point clouds on a known minimal surface (e.g. a discretized Enneper or helicoid patch — analytically known minimal surfaces) so ground truth is available.
- Build the k-NN graph and Laplacian $L_F$ from the same points using your `arrowspace` wiring logic (or a PyTorch/`torch_geometric` reimplementation for prototyping).

**2. Baseline 3D model**

- Standard score network $s_\theta: \mathbb R^3 \times [0,T] \to \mathbb R^3$ (small MLP or PointNet-style), trained with denoising score matching directly on the 3D coordinates.[^1]
- Sample via the reverse SDE Euler–Maruyama update above.

**3. 2.5-D dual-space model**

- Encoder: project 3D points to a 2D item-coordinate $z$ (e.g. via the graph's leading Laplacian eigenvectors / spectral embedding) plus a feature channel $f$ (Rayleigh/λτ score per node).
- Score network $s_\theta^{2.5D}: (\mathbb R^2, f) \times [0,T] \to \mathbb R^2$, where $f$'s evolution is a fixed (non-learned) sparse Laplacian smoothing step, not part of backprop through a determinant.
- Decoder: reconstruct approximate 3D coordinates from $(z, f)$ for evaluation only (e.g. via a fixed local-frame lift or a small learned decoder head).

**4. Training loop (shared skeleton)**

```python
import torch, torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim))
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

def dsm_loss(score_net, x0, sigma_fn, T=1.0):
    t = torch.rand(x0.shape[^0], 1) * T
    sigma = sigma_fn(t)
    noise = torch.randn_like(x0)
    xt = x0 + sigma * noise
    target = -noise / sigma
    pred = score_net(xt, t)
    return ((pred - target) ** 2).sum(-1).mean()
```

Train `ScoreNet(dim=3)` on raw coordinates for the baseline, and `ScoreNet(dim=2)` on $z$ (with $f$ injected as a conditioning feature, e.g. concatenated to the input) for the 2.5-D model.

**5. Sampling**

- Implement Euler–Maruyama for both, using $L_F$ sparse-matrix-vector products (`torch.sparse.mm`) for the feature-channel update in the 2.5-D case — this avoids any dense determinant computation entirely.


## Validation plan

| Metric | What it checks | How computed |
| :-- | :-- | :-- |
| Chamfer distance | Geometric fidelity of generated surface vs. ground truth | Nearest-neighbor distance between generated and reference point sets |
| Rayleigh/λτ energy gap | Whether generated samples preserve the same spectral smoothness as real minimal-surface data | Compare $R(f_{gen})$ vs. $R(f_{real})$ distributions |
| Mean curvature error | Surface-specific quality (does 2.5-D avoid volumetric artifacts while staying minimal-surface-consistent) | Discrete mean curvature estimate on reconstructed mesh vs. analytic minimal surface (should be near zero) |
| Wall-clock / memory | Whether dropping the determinant actually saves compute | Compare training + sampling time and peak memory, 3D vs 2.5-D |
| Spectral stability | Robustness under subsampling, per your Graph Wiring validation protocol | Re-wire graph on random subsamples, measure λτ tail stability [^2] |

**Ablation to isolate the core hypothesis**: train the 2.5-D model with (a) full $(z,f)$ dual space, and (b) $z$ alone (no feature channel), then compare Chamfer distance and curvature error against the 3D baseline. If (a) closes most of the gap to 3D while (b) does not, that's direct evidence that the feature-space Laplacian information — not just a 2D coordinate — is what substitutes for the missing third dimension, which is exactly the "reunification of dual space" claim in ESDM's design doc.
