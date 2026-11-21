"""
ESDM -- Entropic Semantic Diffusion Model.

Unified package containing:
  - The entropic-clock ESDM (clock, vibrational state, ELBO)
  - The 2.5-D dual-space score track (score3d, score25d, minimal surfaces)
  - Vendored VDT upstream code in ``esdm._vendor`` (verbatim, commit 55cec39)

The v2 entropic-clock track reconstructs free-amplitude states using
bounded gains in a frozen graph chart.

"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("esdm")
except PackageNotFoundError:
    __version__ = "2.0.1"

__all__ = ["__version__"]
