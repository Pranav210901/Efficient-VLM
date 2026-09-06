#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
source scripts/alignment_v4_wave0_partition.sh
alignment_v4_wave0_partition_profile teaching 7
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: evaluation-only array 0-6 for accepted Slurm array 2223439 checkpoints."
  exit 0
fi

job_id="$(
  timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
    sbatch --parsable --partition=teaching --array="0-6%7" \
    --job-name=a4w0_accepted_eval \
    --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
    --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-accepted" \
    "${ALIGNMENT_WAVE0_GPU_SBATCH}"
)"
echo "Submitted evaluation only: ${job_id}. No training command was submitted."
