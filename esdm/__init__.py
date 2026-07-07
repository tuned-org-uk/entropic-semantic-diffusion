"""
ESDM — 2.5-D dual-space diffusion track.

New track (docs/000-DIFFUSION-ON-PHYSICAL-SURFACES.md,
001-IMPLEMENTATION-PLAN.md, 002-EXPERIMENTS.md). The legacy VDT +
entropic-clock architecture is preserved as ``esdm_legacy``.

All graph/Laplacian/Rayleigh-quotient computation goes through the adapter in
``esdm.graphs.wiring`` which wraps the real ``arrowspace`` Rust bindings.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("esdm")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

__all__ = ["__version__"]
