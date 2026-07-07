"""
esdm-build  --  one-shot script: load corpus embeddings V, run ArrowSpace GW
                to produce L_F, then compute and save the item-space fused
                matrix M_N = alpha*norm(VV^T) - beta*norm(VL_FV^T).

Usage (via uv):
    uv run esdm-build --v_path data/V.pt --out_dir data/ --alpha 0.5 --beta 0.5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build item-space fused matrix M_N")
    parser.add_argument("--v_path",   required=True, help="Path to corpus embeddings V (.pt)")
    parser.add_argument("--lf_path",  default=None,  help="Path to pre-built L_F (.pt). If omitted, computed from V.")
    parser.add_argument("--out_dir",  default="data", help="Directory to write L_F.pt and M_N.pt")
    parser.add_argument("--alpha",    type=float, default=0.5)
    parser.add_argument("--beta",     type=float, default=0.5)
    parser.add_argument("--n_eigvecs",type=int,   default=64, help="Top-k eigenvectors to cache")
    args = parser.parse_args()

    import torch
    # DualSpaceMatrix import (uncomment when Phase 1 is implemented)
    # from ..laplacian import DualSpaceMatrix

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    V = torch.load(args.v_path, weights_only=True)
    logger.info("Loaded V: %s", V.shape)

    if args.lf_path:
        L_F = torch.load(args.lf_path, weights_only=True)
        logger.info("Loaded L_F: %s", L_F.shape)
    else:
        # Placeholder: replace with actual ArrowSpace GW call
        # from vdeductive.laplacian import build_gw_laplacian
        # L_F = build_gw_laplacian(V.T)  # GW operates on F×N transposed
        raise NotImplementedError(
            "Pass --lf_path or implement ArrowSpace GW call in build_laplacian.py"
        )

    torch.save(L_F, out / "L_F.pt")
    logger.info("Saved L_F -> %s", out / "L_F.pt")

    # dsm = DualSpaceMatrix(alpha=args.alpha, beta=args.beta)
    # dsm.build(V, L_F)
    # torch.save({"M": dsm.M, "eigvals": dsm.eigvals, "eigvecs": dsm.eigvecs}, out / "M_N.pt")
    # logger.info("Saved M_N -> %s", out / "M_N.pt")
    logger.info("build_laplacian: Phase 1 not yet implemented. Stub complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
