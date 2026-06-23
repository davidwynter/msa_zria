#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${WWKG_BASE_URL:?WWKG_BASE_URL must be set}"
: "${WWKG_WORKSPACE:?WWKG_WORKSPACE must be set}"
: "${WWKG_BRANCH:?WWKG_BRANCH must be set}"

export MSA_ZRIA_RUN_WWKG_LIVE=1
PYTHONPATH="src:../contextkg/src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest discover -s tests -p "test_wwkg_live.py" -v
