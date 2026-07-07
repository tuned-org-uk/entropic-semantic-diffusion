"""
ESDM -- Entropic Semantic Diffusion Model.

Unified package containing:
  - The entropic-clock ESDM (clock, vibrational state, WHT filter, ELBO)
  - The 2.5-D dual-space score track (score3d, score25d, minimal surfaces)
  - Vendored VDT upstream code in ``esdm._vendor`` (verbatim, commit 55cec39)

All graph/Laplacian/Rayleigh-quotient computation routes through the
``arrowspace`` adapter in ``esdm.graphs.wiring`` (with graceful fallback
when the Rust bindings are absent).

The WHT spectral filter is an optional, injectable component: pass a
``GraphWHTFilter`` instance to ``ESDM(cfg)`` via ``cfg["model"]["wht_filter"]``
or ``None`` to disable. The CLI surfaces this as ``--wht-filter {none,basic,bivariate}``.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("esdm")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

__all__ = ["__version__"]
