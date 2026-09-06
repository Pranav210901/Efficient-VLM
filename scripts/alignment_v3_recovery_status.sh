#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner status \
  --pipeline configs/alignment_v3_recovery/pipeline.yaml
if command -v squeue >/dev/null 2>&1; then
  squeue --me || true
fi

