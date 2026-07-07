"""
WHT filter factory for the CLI.

Translates the --wht-filter {none,basic,bivariate} flag into a
GraphWHTFilter instance (or None) that can be injected into the ESDM model.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from esdm.graphs.wht_filter import GraphWHTFilter


WHT_FILTER_CHOICES = ("none", "basic", "bivariate")


def build_wht_filter(
    choice: str,
    L_F: np.ndarray,
    a_tau_scale: float = 5.0,
    tau_decay_power: float = 1.0,
) -> Optional[GraphWHTFilter]:
    """
    Construct a GraphWHTFilter from the CLI choice string.

    Parameters
    ----------
    choice : str
        One of "none", "basic", "bivariate".
        - "none":     no filter (bare L_f @ Q in the wave recurrence)
        - "basic":    heat-kernel-only filter (no bivariate gate)
        - "bivariate": Path A bivariate gate g_k(tau, r_k) with heat-death closure
    L_F : np.ndarray
        Feature-space Laplacian (F, F), symmetric PSD.
    a_tau_scale : float
        Scale parameter for the bivariate gate (only used by "bivariate").
    tau_decay_power : float
        Power parameter for the bivariate gate (only used by "bivariate").
    """
    if choice == "none":
        return None
    if choice not in WHT_FILTER_CHOICES:
        raise ValueError(
            f"Unknown wht_filter choice '{choice}'. "
            f"Must be one of {WHT_FILTER_CHOICES}"
        )
    filt = GraphWHTFilter(L_matrix=L_F)
    filt.fit()
    if choice == "bivariate":
        filt.a_tau_scale = a_tau_scale
        filt.tau_decay_power = tau_decay_power
    return filt
