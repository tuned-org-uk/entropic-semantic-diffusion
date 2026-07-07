"""
ESDM graph primitives.

  - WireGraph: arrowspace adapter for graph construction + Laplacian + Rayleigh
  - DualSpaceMatrix: fused item-space matrix from corpus embeddings
  - GraphWHTFilter: Walsh-Hadamard spectral filter (optional, injectable)
"""
from .wiring import WireGraph
from .laplacian import DualSpaceMatrix
from .wht_filter import GraphWHTFilter, heat_kernel

__all__ = ["WireGraph", "DualSpaceMatrix", "GraphWHTFilter", "heat_kernel"]
