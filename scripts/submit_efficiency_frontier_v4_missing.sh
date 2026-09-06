#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/efficiency_frontier/pipeline.yaml"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.efficiency_frontier validate --pipeline "${PIPELINE}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: train historical-v4 seeds 43/44 as a 2-task teaching array."
  echo "DRY-RUN: evaluate frontier job indices 7/8 after both training tasks succeed."
  echo "DRY-RUN: run validation after evaluation; seed 42 is never retrained."
  exit 0
fi

train_id="$(timeout --foreground 60 sbatch --parsable \
  --array=0-1%2 --job-name=ef_v4_train \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=train-missing-v4" \
  slurm/efficiency_frontier/gpu_train.sbatch)"
eval_id="$(timeout --foreground 60 sbatch --parsable \
  --array=7-8%2 --dependency="afterok:${train_id}" --job-name=ef_v4_eval \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=evaluate" \
  slurm/efficiency_frontier/gpu_eval.sbatch)"
audit_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${eval_id}" --job-name=ef_v4_audit \
  --export="ALL,EFFICIENCY_FRONTIER_ROOT=${ROOT},EFFICIENCY_FRONTIER_COMMAND=validate" \
  slurm/efficiency_frontier/cpu_report.sbatch)"
echo "Historical v4 completion: train=${train_id} eval=${eval_id} audit=${audit_id}"
