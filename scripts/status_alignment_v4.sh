#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
for teacher in mobileclip2 siglip2; do
  echo "===== ${teacher} ====="
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner status \
    --pipeline "configs/alignment_v4/pipeline_${teacher}.yaml"
done
echo "===== Slurm ====="
squeue -u "${USER}" -o "%.18i %.9P %.28j %.2t %.10M %.30R" || true

