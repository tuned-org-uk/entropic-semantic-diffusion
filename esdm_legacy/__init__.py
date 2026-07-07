"""
ESDM legacy track — VDT + entropic-clock + WHT-filter architecture.

This package is the pre-2.5-D implementation, preserved verbatim under the
``esdm_legacy`` import name so the legacy experiments (Exp 1-3) and their
tests keep running while the new 2.5-D dual-space track is developed under
``esdm/``. See docs/legacy/01-IMPLEMENTATION-PLAN.md for the legacy design
and docs/001-IMPLEMENTATION-PLAN.md for the new track that replaces it.

Internal imports are relative so the package is relocatable.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("esdm")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

# --- Legacy ESDM modules (stub re-exports; kept for reference) ---
# from .laplacian          import DualSpaceMatrix
# from .clock              import EntropicClock, ClockState
# from .vibrational        import VibrationalStateBlock
# from .diffusion_decoder  import EntropicDiffusionDecoder
# from .elbo               import esdm_elbo
# from .model              import ESDM, ESDMOutput

__all__ = ["__version__"]
