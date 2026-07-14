"""Pytest configuration for ESDM."""
from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., frozen DINO encoding downloads).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow (deselected by default)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow") or os.environ.get("ESDM_RUN_SLOW", "0") == "1":
        return
    skip_slow = pytest.mark.skip(reason="use --run-slow or set ESDM_RUN_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
