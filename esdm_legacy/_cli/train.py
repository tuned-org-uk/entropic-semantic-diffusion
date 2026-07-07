"""
esdm-train  --  training entry point for the ESDM.

Usage (via uv):
    uv run esdm-train --config esdm/_configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESDM")
    parser.add_argument("--config", default="esdm/_configs/default.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    # Stub: uncomment as phases are implemented
    # import torch
    # from ..model import ESDM
    # model = ESDM(cfg)
    # ... training loop ...
    logger.info("esdm-train: model not yet instantiated. Implement esdm/model.py first.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
