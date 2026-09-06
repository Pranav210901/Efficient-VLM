#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PARTITION=teaching
MAX_GPUS=8
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PARTITION}" == teaching ]] || { echo 'FreezeShift requires teaching/native-BF16 RTX PRO 6000 hardware.' >&2; exit 2; }
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || { echo '--max-total-gpus must be in [1,8].' >&2; exit 2; }
mkdir -p freezeshift/{logs/slurm,results/manifests,results/report,checkpoints}
"${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner validate
submit() {
  if [[ "${DRY_RUN}" == true ]]; then echo "DRY-RUN sbatch $*" >&2; echo "97${RANDOM}"; else timeout --foreground 60 sbatch --parsable "$@"; fi
}
smoke="$(submit --partition="${PARTITION}" --job-name=fs_smoke --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/smoke.sbatch)"
train="$(submit --partition="${PARTITION}" --job-name=fs_train --array="0-8%${MAX_GPUS}" --dependency="afterok:${smoke}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/train.sbatch)"
verify="$(submit --partition="${PARTITION}" --job-name=fs_verify --array="0-2%3" --dependency="afterok:${train}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/verify.sbatch)"
evaluate="$(submit --partition="${PARTITION}" --job-name=fs_eval --array="0-215%${MAX_GPUS}" --dependency="afterok:${verify}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/evaluate.sbatch)"
arm_report="$(submit --partition="${PARTITION}" --job-name=fs_arm_report --array="0-2%3" --dependency="afterok:${evaluate}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/report_arm.sbatch)"
profile="$(submit --partition="${PARTITION}" --job-name=fs_profile --dependency="afterok:${arm_report}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/profile.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=fs_report --dependency="afterok:${profile}" --export="ALL,FREEZESHIFT_ROOT=${ROOT}" freezeshift/slurm/report.sbatch)"
echo "FreezeShift: smoke=${smoke} train=${train} verify=${verify} eval=${evaluate} arm_report=${arm_report} profile=${profile} report=${report}"
echo 'Nine training runs: vision-only, text-only and dual-tower LoRA x three seeds.'
echo 'Flickr test remains sealed. Codex did not submit jobs.'
