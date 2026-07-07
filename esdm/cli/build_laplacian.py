"""
ESDM Laplacian builder CLI.

Builds a feature-space Laplacian from corpus embeddings and saves it as a
.pt file for use by the training/sweep CLIs.

Usage:
    esdm-build --features data/V.pt --output data/L_F.pt --k 16
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from esdm.graphs.wiring import WireGraph

logger = logging.getLogger("esdm.cli.build_laplacian")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="esdm-build",
        description="Build a feature-space Laplacian from corpus embeddings.",
    )
    parser.add_argument("--features", "-f", required=True, help="Path to V.pt (N, D)")
    parser.add_argument("--output", "-o", default="data/L_F.pt", help="Output path")
    parser.add_argument("--k", type=int, default=16, help="k-NN neighbors")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)

    V = torch.load(Path(args.features))
    graph = WireGraph.build(V, k=args.k, seed=args.seed)
    L = graph.laplacian(normalized=True, sparse=False)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(L, Path(args.output))
    logger.info("Saved Laplacian shape=%s nnz=%d -> %s",
                tuple(L.shape), int((L != 0).sum()), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
