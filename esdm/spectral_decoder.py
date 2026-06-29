"""
Phase 3 (adapter) — SpectralLoadingDecoder.

Thin re-export of the VDT SpectralLoadingDecoder. The interface is unchanged;
this file exists so the rest of the ESDM package has a single import surface.
"""
from __future__ import annotations

# from vdeductive.spectral import SpectralLoadingDecoder  # uncomment when vdeductive is installed
# from vdeductive.wiring_decoder import WiringDecoder

# Placeholder types so the rest of the package can import without error
class SpectralLoadingDecoder:
    """Stub — replace with: from vdeductive.spectral import SpectralLoadingDecoder"""
    def __init__(self, *args, **kwargs):  # noqa: ANN001
        raise NotImplementedError("Import SpectralLoadingDecoder from vdeductive.spectral")
