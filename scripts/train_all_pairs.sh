#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" scripts/run_encoder_matrix.py --config "${1:-configs/experiment_blf.yaml}" --mode all
