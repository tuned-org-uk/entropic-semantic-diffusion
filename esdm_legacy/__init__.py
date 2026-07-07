"""
ESDM — Entropic Semantic Diffusion Model

Public re-exports for the most commonly used classes and functions.
All heavy VDT primitives are imported from `vdeductive` directly;
this package provides only the ESDM-specific modules on top.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("esdm")
except PackageNotFoundError:  # editable install without metadata
    __version__ = "0.1.0.dev"

# --- ESDM-specific modules (stub imports; filled in as phases are implemented) ---
# from esdm.laplacian import DualSpaceMatrix
# from esdm.clock     import EntropicClock, ClockState
# from esdm.vibrational import VibrationalStateBlock  # ESDM-modified
# from esdm.diffusion_decoder import EntropicDiffusionDecoder
# from esdm.elbo      import esdm_elbo
# from esdm.model     import ESDM, ESDMOutput

__all__ = ["__version__"]
