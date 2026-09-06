#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/cc3m_scale/pipeline.yaml"

if [[ ! -f data/cc3m_v1_1/manifests/acquisition_complete.json ]]; then
  echo "BLOCKED: CC3M acquisition is incomplete." >&2
  exit 2
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: tar-index build/validate -> one native-BF16 GPU batch smoke."
  echo "DRY-RUN: no Arm A/B training and no Flickr test evaluation."
  exit 0
fi
index_id="$(timeout --foreground 60 sbatch --parsable \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/build_index.sbatch)"
smoke_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${index_id}" \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/gpu_smoke.sbatch)"
echo "CC3M training smoke: index=${index_id} smoke=${smoke_id}"
