#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner status --pipeline "${ALIGNMENT_V3_PIPELINE:-configs/alignment_v3/pipeline.yaml}"
if command -v squeue >/dev/null 2>&1; then squeue --me || true; fi

