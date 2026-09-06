#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -n "${ALIGNMENT_V3_SYSTEM_PYTHON:-}" ]]; then
  PYTHON="${ALIGNMENT_V3_SYSTEM_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${ROOT}/.venv-aisurrey/bin/python"
fi
echo "Queue factorial submit Python: ${PYTHON}"
"${PYTHON}" -m src.alignment_v3.queue_factorial validate
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: 2 teaching/native-BF16 smoke tasks, 1 GPU/12 CPU/96G/8h each."
  exit 0
fi
job_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --gres=gpu:nvidia_rtxpro6000:1 \
  --array=0-1%2 --job-name=qf_smoke \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=smoke" \
  slurm/queue_factorial/gpu_stage.sbatch)"
report_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${job_id}" --job-name=qf_smoke_report \
  --export="ALL,QUEUE_FACTORIAL_ROOT=${ROOT},QUEUE_FACTORIAL_COMMAND=smoke-report" \
  slurm/queue_factorial/cpu_stage.sbatch)"
echo "Queue-factorial smoke: train=${job_id} report=${report_id}"
