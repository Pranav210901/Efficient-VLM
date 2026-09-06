#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${ALIGNMENT_V3_PYTHON}" ]]; then
  echo "BLOCKED: expected Python is not executable: ${ALIGNMENT_V3_PYTHON}" >&2
  exit 2
fi
export ALIGNMENT_V3_PYTHON
source scripts/alignment_v4_wave0_partition.sh
alignment_v4_wave0_partition_profile teaching 2
# The all-caption batch-1024 loader uses ten persistent workers with four
# prefetched batches each. Both 48G confirmation tasks were killed by the
# Slurm host-memory cgroup (job 2279494), while GPU usage stayed below 7G.
# Raising only the Slurm host-memory allocation preserves the saved config
# fingerprint and allows exact checkpoint resume.
ALIGNMENT_WAVE0_GPU_MEM="96G"
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
ANCHOR="results/alignment_v4_wave0/wave0-lr-control/infonce_no_queue__captions_all__b1024__lr_3p0__seed_42/metrics.json"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: teaching/native-BF16 seeds 43-44, host_mem=${ALIGNMENT_WAVE0_GPU_MEM}, checkpoint resume, evaluation, then conditional lock."
  exit 0
fi
if [[ ! -f "${ANCHOR}" ]]; then
  echo "BLOCKED: evaluate accepted index 0 first; missing ${ANCHOR}" >&2
  exit 2
fi
# PENDING ledger entries already exist from the original training attempt.
# Avoid importing the full ML stack on the login node merely to de-duplicate
# them; the evaluation stage closes each exact fingerprint with COMPLETE.
train_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-1%2" --job-name=a4w0_confirm_train \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=train-wave0-winner-confirmation" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"
eval_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --array="0-1%2" --dependency="afterok:${train_id}" --job-name=a4w0_confirm_eval \
  --cpus-per-task="${ALIGNMENT_WAVE0_GPU_CPUS}" --mem="${ALIGNMENT_WAVE0_GPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=eval-wave0-winner-confirmation" \
  "${ALIGNMENT_WAVE0_GPU_SBATCH}")"
lock_id="$(timeout --foreground 60 sbatch --parsable --partition=teaching \
  --dependency="afterok:${eval_id}" --job-name=a4w0_recipe_lock \
  --cpus-per-task="${ALIGNMENT_WAVE0_CPU_CPUS}" --mem="${ALIGNMENT_WAVE0_CPU_MEM}" \
  --export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=lock-wave0-corrected-recipe" \
  "${ALIGNMENT_WAVE0_CPU_SBATCH}")"
echo "Confirmation: train=${train_id} eval=${eval_id} conditional_lock=${lock_id}"
