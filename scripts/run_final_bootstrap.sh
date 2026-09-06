#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.final_bootstrap --iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed 20260828
