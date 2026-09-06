#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" scripts/train.py --config configs/sanity.yaml
