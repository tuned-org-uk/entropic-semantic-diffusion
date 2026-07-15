"""
ESDM graph primitives.

  - WireGraph: arrowspace adapter for graph construction + Laplacian + Rayleigh
  - DualSpaceMatrix: fused item-space matrix from corpus embeddings
"""
from .wiring import WireGraph
from .laplacian import DualSpaceMatrix

__all__ = ["WireGraph", "DualSpaceMatrix"]
