"""
Phase 3 (adapter) — SpectralLoadingDecoder.

Thin re-export of the VDT ``SpectralLoadingDecoder`` from
``esdm._vendor.wiring_decoder``. The original implementation lives in the
vendored `_vendor` package; this module exists so the rest of the ESDM
package has a single import surface and can be replaced by an ESDM-specific
wrapper later without changing call-sites.

The VDT forward signature is:

    forward(z: Tensor, U_q: Tensor, L_base: Tensor)
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

returning ``(W, omega, S, L_z, log_var_S)``.  ESDM-specific quantities such as
``N_active``, ``log_lam_s``, ``kl_S``, ``kl_tau`` and ``dark_entropy`` are
derived from these tensors in the top-level ``ESDM`` model.
"""
from __future__ import annotations

from .._vendor.wiring_decoder import SpectralLoadingDecoder

__all__ = ["SpectralLoadingDecoder"]
