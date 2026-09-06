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
  echo "DRY-RUN: build/verify tar index -> six training tasks (adaptive concurrency) -> report."
  echo "DRY-RUN: three Arm A and three Arm B; Flickr test evaluations=0."
  exit 0
fi

read -r max_tasks eligible_nodes < <(
  sinfo -h -p teaching -N -o '%N|%G|%c' |
  awk -F'|' '
    /gpu:nvidia_rtxpro6000:/ {
      gpus=0
      if (match($2, /gpu:nvidia_rtxpro6000:([0-9]+)/, a)) gpus=a[1]+0
      cpu_tasks=int(($3+0)/12)
      slots=(gpus < cpu_tasks ? gpus : cpu_tasks)
      total+=slots
      nodes+=1
    }
    END { print total+0, nodes+0 }
  '
)
if [[ "${max_tasks}" -lt 1 ]]; then
  echo "BLOCKED: no teaching RTX PRO 6000 slots compatible with 12 CPUs/task." >&2
  exit 2
fi
concurrency=$((max_tasks < 6 ? max_tasks : 6))

index_id="$(timeout --foreground 60 sbatch --parsable \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/build_index.sbatch)"
train_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${index_id}" --array="0-5%${concurrency}" \
  --job-name=cc3m_train \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/gpu_train.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${train_id}" \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
  slurm/cc3m_scale/training_report.sbatch)"
echo "CC3M training: index=${index_id} train=${train_id} report=${report_id}"
