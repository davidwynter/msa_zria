#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m compileall src tests
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest discover -s tests -v
