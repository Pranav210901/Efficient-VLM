#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v4_wave0_partition.sh

PARALLEL_TASKS=6
PER_TASK_MEMORY_GB=96
alignment_v4_wave0_partition_profile teaching "${PARALLEL_TASKS}"
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${ALIGNMENT_V3_PYTHON}" ]]; then
  echo "BLOCKED: expected Python is not executable: ${ALIGNMENT_V3_PYTHON}" >&2
  exit 2
fi

"${ALIGNMENT_V3_PYTHON}" - <<'PY'
import json
from pathlib import Path

recipe = json.loads(Path(
    "results/alignment_v4_wave0/selection/recipe.json"
).read_text())
rescreen = json.loads(Path(
    "results/alignment_v4_wave0/wave0-rescreen/report.json"
).read_text())
if recipe.get("status") != "SELECTED_WAVE0_RECIPE":
    raise SystemExit("BLOCKED: Wave 0 recipe is not locked.")
if rescreen.get("status") != "RANKING_CHANGED_STOP_FOR_DECISION":
    raise SystemExit("BLOCKED: ranking-change report is not closed.")
if rescreen.get("locked_student") != "dinov3_vits16__all_minilm_l6_v2":
    raise SystemExit("BLOCKED: MiniLM student lock is missing.")
if rescreen.get("student_pair_changed") is not False:
    raise SystemExit("BLOCKED: student pair must not change before confirmation.")
PY

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: independent one-GPU arrays; up to ${PARALLEL_TASKS} concurrent tasks, cpus_per_task=${ALIGNMENT_WAVE0_GPU_CPUS}, mem_per_task=${PER_TASK_MEMORY_GB}G."
  echo "Graph: 6-task training array -> 6-task evaluation array -> report."
  exit 0
fi

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending-stage \
  --pipeline "${PIPELINE}" --ledger-stage wave0-rescreen-confirmation

train_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-5%${PARALLEL_TASKS}" --job-name=a4w0_top3_train \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" \
  --mem="${PER_TASK_MEMORY_GB}G" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_V3_PYTHON=${ALIGNMENT_V3_PYTHON},ALIGNMENT_WAVE0_COMMAND=train-wave0-rescreen-confirmation" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"

eval_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-5%${PARALLEL_TASKS}" --dependency="afterok:${train_id}" \
  --job-name=a4w0_top3_eval \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" \
  --mem="${PER_TASK_MEMORY_GB}G" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_V3_PYTHON=${ALIGNMENT_V3_PYTHON},ALIGNMENT_WAVE0_COMMAND=eval-wave0-rescreen-confirmation" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"

report_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${eval_id}" --job-name=a4w0_top3_report \
  --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" \
  --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_V3_PYTHON=${ALIGNMENT_V3_PYTHON},ALIGNMENT_WAVE0_COMMAND=report-wave0-rescreen-confirmation" \
  "${ALIGNMENT_WAVE0_CPU_SBATCH}")"

echo "Top-three confirmation: train=${train_id} eval=${eval_id} report=${report_id}"
