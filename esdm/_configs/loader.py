"""
Dataset configuration loader.
Reads per-dataset YAML configs from esdm/_configs/dataset_*.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml


_CONFIGS_DIR = Path(__file__).parent


def load_dataset_config(dataset_name: str) -> dict:
    """Load the per-dataset YAML config.

    Returns a dict with keys: dataset, graph_params, training.
    Raises FileNotFoundError if no config exists for the dataset.
    """
    path = _CONFIGS_DIR / f"dataset_{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset config found at {path}. "
            f"Create one with graph_params from arrowspace_tuner."
        )
    with open(path) as f:
        return yaml.safe_load(f)
