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
[[ "${PARTITION}" == teaching ]] || { echo 'Final LR study requires teaching/native-BF16 hardware.' >&2; exit 2; }
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || { echo '--max-total-gpus must be in [1,8].' >&2; exit 2; }
mkdir -p logs/final_lr_study/slurm results/final_lr_study/{manifests,selection,report}
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.final_lr_study validate
submit() {
  if [[ "${DRY_RUN}" == true ]]; then echo "DRY-RUN sbatch $*" >&2; echo "98${RANDOM}"; else timeout --foreground 60 sbatch --parsable "$@"; fi
}
screen_train="$(submit --partition="${PARTITION}" --job-name=flr_screen_tr --array="0-3%${MAX_GPUS}" --export="ALL,FINAL_LR_ROOT=${ROOT}" slurm/final_lr_study/screen_train.sbatch)"
screen_eval="$(submit --partition="${PARTITION}" --job-name=flr_screen_ev --array="0-95%${MAX_GPUS}" --dependency="afterok:${screen_train}" --export="ALL,FINAL_LR_ROOT=${ROOT}" slurm/final_lr_study/screen_evaluate.sbatch)"
selection="$(submit --partition="${PARTITION}" --job-name=flr_select --dependency="afterok:${screen_eval}" --export="ALL,FINAL_LR_ROOT=${ROOT},FINAL_LR_COMMAND=select" slurm/final_lr_study/control.sbatch)"
confirm_train="$(submit --partition="${PARTITION}" --job-name=flr_confirm_tr --array="0-1%${MAX_GPUS}" --dependency="afterok:${selection}" --export="ALL,FINAL_LR_ROOT=${ROOT}" slurm/final_lr_study/confirm_train.sbatch)"
confirm_eval="$(submit --partition="${PARTITION}" --job-name=flr_confirm_ev --array="0-47%${MAX_GPUS}" --dependency="afterok:${confirm_train}" --export="ALL,FINAL_LR_ROOT=${ROOT}" slurm/final_lr_study/confirm_evaluate.sbatch)"
report="$(submit --partition="${PARTITION}" --job-name=flr_report --dependency="afterok:${confirm_eval}" --export="ALL,FINAL_LR_ROOT=${ROOT},FINAL_LR_COMMAND=report" slurm/final_lr_study/control.sbatch)"
echo "Final LR study: screen_train=${screen_train} screen_eval=${screen_eval} select=${selection} confirm_train=${confirm_train} confirm_eval=${confirm_eval} report=${report}"
echo 'Maximum new training runs: 6 (4 screen + 2 confirmation); current LR is reused.'
echo 'Flickr30k test remains sealed. No jobs were submitted by Codex.'
