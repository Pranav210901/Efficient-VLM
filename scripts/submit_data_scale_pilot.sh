#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/data_scale_pilot/pipeline.yaml"

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.data_scale_pilot prepare --pipeline "${PIPELINE}" >/dev/null
echo "Data-scale manifests: prepared or reused with provenance verification."
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.data_scale_pilot validate --pipeline "${PIPELINE}"

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
concurrency=$((max_tasks < 7 ? max_tasks : 7))
echo "Teaching eligible nodes=${eligible_nodes}; pilot concurrency=${concurrency}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: 7 unique training jobs -> 7 validation evaluations -> report."
  echo "DRY-RUN: Flickr30k test evaluation count is zero."
  exit 0
fi
train_id="$(timeout --foreground 60 sbatch --parsable \
  --array="0-6%${concurrency}" --job-name=ds_train \
  --export="ALL,DATA_SCALE_ROOT=${ROOT},DATA_SCALE_COMMAND=train" \
  slurm/data_scale_pilot/gpu_stage.sbatch)"
eval_id="$(timeout --foreground 60 sbatch --parsable \
  --array="0-6%${concurrency}" --dependency="afterok:${train_id}" --job-name=ds_eval \
  --export="ALL,DATA_SCALE_ROOT=${ROOT},DATA_SCALE_COMMAND=evaluate" \
  slurm/data_scale_pilot/gpu_stage.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${eval_id}" --job-name=ds_report \
  --export="ALL,DATA_SCALE_ROOT=${ROOT},DATA_SCALE_COMMAND=report" \
  slurm/data_scale_pilot/cpu_stage.sbatch)"
echo "Data-scale pilot: train=${train_id} eval=${eval_id} report=${report_id}"
