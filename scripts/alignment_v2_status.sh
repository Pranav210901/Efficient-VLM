#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/alignment_v2_env.sh"
alignment_v2_resolve_python
"${ALIGNMENT_V2_PYTHON}" -m src.alignment_v2.runner status --pipeline "${1:-configs/alignment_v2/pipeline.yaml}"
if command -v squeue >/dev/null 2>&1; then
  squeue --me --name=a2_validate,a2_prefetch,a2_reference,a2_train,a2_evaluate,a2_report
fi
