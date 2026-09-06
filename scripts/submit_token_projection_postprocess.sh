#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PARTITION=teaching
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == "teaching" ]] || {
  echo "Trained-weight profiling requires teaching/native-BF16 hardware." >&2
  exit 2
}

profile="$(timeout --foreground 60 sbatch --parsable \
  --partition="${PARTITION}" --job-name=tp_profile_fix \
  --array="0-2%1" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT}" \
  slurm/token_projection_training/profile.sbatch)"
report="$(timeout --foreground 60 sbatch --parsable \
  --partition="${PARTITION}" --job-name=tp_report_fix \
  --dependency="afterok:${profile}" \
  --export="ALL,TOKEN_TRAIN_ROOT=${ROOT},TOKEN_TRAIN_COMMAND=report" \
  slurm/token_projection_training/cpu.sbatch)"
echo "Token-projection postprocess only: profile=${profile} report=${report}"
echo "No training and no epoch evaluation will run."
