!/bin/bash

set -eux

# Lint
pip install ruff typos
# Type
pip install mypy
