#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv run --no-sync python -m esdm.experiments.representation_search "$@"
