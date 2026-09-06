#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Completed reference, training and evaluation tasks return immediately. Any
# interrupted training task resumes from latest.pt through the normal runner.
exec bash scripts/submit_alignment_v2_pipeline.sh \
  --mode parallel \
  --max-total-gpus "${ALIGNMENT_V2_MAX_TOTAL_GPUS:-8}" \
  --partition "${ALIGNMENT_V2_PARTITION:-teaching}" \
  --pipeline "${ALIGNMENT_V2_PIPELINE:-configs/alignment_v2/pipeline.yaml}" \
  "$@"
