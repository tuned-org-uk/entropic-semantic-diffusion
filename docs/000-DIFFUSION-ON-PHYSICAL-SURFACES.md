Let's find a working model that keeps dimensionality below the 3D so to not deal with volumes and ODE.
 The information contained in the dual space should be enough to keep the results valid avoiding computing determinants (volumes).
 The strong assumption is that the same quality of output of a diffusion model can be maintained without reaching the
 full 3-dimensional representation but sticking to an approximate "2.5-D" unified dual-space representation that collects all
 the information harvested from the item-space and feature-space as constructed by graph wiring.

Since graph wiring already shows that Rayleigh-quotient/Dirichlet energy minimization on a k-NN Laplacian converges to
 Dirichlet-energy minimization on the underlying manifold, and that Dirichlet minimizers in conformal gauge are minimal surfaces
 (i.e., harmonic maps rather than volume-based objects), you can build a diffusion process that operates entirely on graph-spectral
 quantities and never needs a Jacobian determinant or a full 3D flow map [^1][^2].

## Why you can drop the determinant

The determinant/Jacobian machinery in Little Book Of Generative AI Chapter 5's Fokker–Planck derivation exists specifically to
 track how *volume* changes under a flow map in a continuous ambient space [^3]. But Moriondo's graph wiring formalism already replaces
 the ambient manifold with a discrete Laplacian $L_F$ whose Rayleigh quotient $R(x) = x^\top L_F x / x^\top x$ converges to
 the continuum Dirichlet energy $E_{\mathcal M_F}(f) = \tfrac12\int_{\mathcal M_F}\|\nabla_{\mathcal M_F} f\|^2\,dV$ [^2].
 Because Dirichlet energy is a quadratic form on the graph rather than a measure of local volume distortion, tracking its
 evolution under noise never requires a determinant — you only need the Laplacian's spectrum and how mass redistributes across
 eigenmodes [^2].

## Reframing Fokker–Planck for a graph-wired dual space

In the continuous case, Fokker–Planck combines a divergence (transport) term with a Laplacian (diffusion) term:

$$
\partial_t p(x,t) = -\nabla\cdot(p\,v) + \frac{\sigma^2}{2}\Delta p(x,t)
$$

On Moriondo's wired graph, the natural discrete analogue replaces the continuum Laplacian $\Delta$ directly with the
 graph Laplacian $L_F$ (or its normalized form $L_{sym}$), since Theorem 3.1 in Moriondo's Graph Wiring paper already proves this
 operator converges to the Laplace–Beltrami operator under standard sampling assumptions [^2]. This gives a
 **graph Fokker–Planck update** that lives purely in spectral/item-feature dual space:

$$
p_{t+\Delta t} \approx p_t - \Delta t \, \nabla_{\text{item}}\cdot(p_t v_t) + \frac{\sigma^2\Delta t}{2}\, L_F p_t
$$

Here $L_F p_t$ is a sparse matrix–vector product (no determinant, no volume element) — exactly the heat-flow step
 $x \leftarrow x - \eta L x$ your `arrowspace` already implements [^1]. This is the "2.5-D" trick: item-space transport
 (the divergence term, still effectively low-dimensional since it acts on the dual coordinate, not on a 3D embedding)
 plus feature-space diffusion (the Laplacian term, driven entirely by the wired graph's spectrum) [^1][^2].

## Concrete architecture sketch for ESDM

| Component | Continuous 3D analogue | Your graph-native replacement |
| :-- | :-- | :-- |
| State space | $\mathbb{R}^3$ point cloud | Dual (item, feature) graph pair from graph wiring [^2] |
| Volume distortion | Jacobian determinant | Not needed — Rayleigh quotient / Dirichlet energy tracks "spread" instead [^2] |
| Diffusion operator | Laplacian $\Delta$ on ambient space | Graph Laplacian $L_F$ on wired feature manifold [^2] |
| Minimal-surface constraint | Mean-curvature-zero PDE | Conformal-gauge Dirichlet minimizer, i.e. λτ-minimizing embedding [^2] |
| Score/drift | $\nabla_x \log p(x)$ in ambient space | Gradient of λτ energy w.r.t. item coordinates (Rayleigh-quotient gradient) |

The score function itself can be redefined as the gradient of the λτ energy rather than the log-density gradient in ambient space: $s(x) \approx -\nabla_x R(x)$, since minimizing Rayleigh energy is variationally equivalent to following the score toward smoother (lower-Dirichlet-energy) configurations on the wired manifold [^2]. This keeps the entire generative trajectory inside the dual item/feature graph representation, using only sparse Laplacian operators — never requiring you to embed into or integrate an ODE over a literal 3D volume [^1][^2].

## What still needs proof for ESDM specifically

Your Graph Wiring paper is explicit that the "Dirichlet ⇒ minimal surface" chain is rigorous only under conformal-gauge assumptions, and that this gauge condition is currently *imposed* rather than *derived* from the discrete wiring pipeline [^2]. For ESDM to inherit "the same quality of output" as a full 3D diffusion model, you'll need to show empirically (per your own validation protocol: spectral stability under subsampling, λτ tail-quality metrics) that the dual-space diffusion trajectory preserves the same low-frequency geometry that a volumetric diffusion model would recover, since that is exactly the open gap your paper flags between "structured assumption" and "unconditional theorem" [^2].

Let's sketch the concrete update equations for a coupled item-space/feature-space diffusion step (i.e., the discrete-time algorithm you'd actually implement in ArrowSpace/ESDM), and also focus first on how to define and validate the "2.5-D" score function empirically.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.tuned.org.uk/posts/014_arrowspace_and_string_theory_minimal_surfaces

[^2]: https://www.tuned.org.uk/assets/graph_wiring.pdf

[^3]: LittleBookOfGenerative.pdf

[^4]: https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220780.02840438

[^5]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Jeschke09b.pdf

[^6]: https://www.math.hkbu.edu.hk/~lling/PDF/GLapSurfacePDEs2017.pdf

[^7]: https://www.authorea.com/doi/pdf/10.22541/au.177430060.02394540

[^8]: https://www.math.utah.edu/~mayer/math/Mayer07.pdf

[^9]: https://www.mendeley.com/catalogue/4503c876-2eca-3d7b-9fab-f26b1201a597/

[^10]: https://www.i2m.univ-amu.fr/fvca5/articles/article55215782_67-CALHOUN_donna_v1.pdf

[^11]: https://www.fiq.unl.edu.ar/depto-mate/pmorin/Papers/13/SINUM-sd-graphs.pdf

[^12]: https://arxiv.org/pdf/1905.06395.pdf

[^13]: https://arxiv.org/html/2510.23757v1

[^14]: https://www.linkedin.com/posts/lorenzomoriondo_arrowspace-performance-analysis-100k-dimensions-activity-7428054002527006720-3uFq

[^15]: https://www.cs.jhu.edu/~misha/Fall09/Belkin08.pdf

[^16]: https://www.tuned.org.uk/arrowspace-paper

