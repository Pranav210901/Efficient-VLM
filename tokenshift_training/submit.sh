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
[[ "${PARTITION}" == teaching ]] || {
  echo 'TokenShift training requires teaching/native-BF16 RTX PRO 6000 hardware.' >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-8]$ ]] || {
  echo '--max-total-gpus must be an integer in [1,8].' >&2
  exit 2
}

mkdir -p tokenshift_training/{logs/slurm,results/manifests,results/profile,results/report,checkpoints}
"${ALIGNMENT_V3_PYTHON}" -m tokenshift_training.runner validate

submit() {
  if [[ "${DRY_RUN}" == true ]]; then
    echo "DRY-RUN sbatch $*" >&2
    case " $* " in
      *" --job-name=tst_smoke "*) echo 97001 ;;
      *" --job-name=tst_train "*) echo 97002 ;;
      *" --job-name=tst_verify "*) echo 97003 ;;
      *" --job-name=tst_eval "*) echo 97004 ;;
      *" --job-name=tst_arm_report "*) echo 97005 ;;
      *" --job-name=tst_profile "*) echo 97006 ;;
      *" --job-name=tst_report "*) echo 97007 ;;
      *) echo 'Unknown dry-run stage' >&2; return 2 ;;
    esac
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

common=(--partition="${PARTITION}" --export="ALL,TOKENSHIFT_TRAINING_ROOT=${ROOT}")
smoke="$(submit "${common[@]}" --job-name=tst_smoke tokenshift_training/slurm/smoke.sbatch)"
train="$(submit "${common[@]}" --job-name=tst_train --array="0-11%${MAX_GPUS}" \
  --dependency="afterok:${smoke}" tokenshift_training/slurm/train.sbatch)"
verify="$(submit "${common[@]}" --job-name=tst_verify --array="0-3%4" \
  --dependency="afterok:${train}" tokenshift_training/slurm/verify.sbatch)"
evaluate="$(submit "${common[@]}" --job-name=tst_eval --array="0-11%${MAX_GPUS}" \
  --dependency="afterok:${verify}" tokenshift_training/slurm/evaluate_seed.sbatch)"
arm_report="$(submit "${common[@]}" --job-name=tst_arm_report --array="0-3%4" \
  --dependency="afterok:${evaluate}" tokenshift_training/slurm/report_arm.sbatch)"
profile="$(submit "${common[@]}" --job-name=tst_profile \
  --dependency="afterok:${arm_report}" tokenshift_training/slurm/profile.sbatch)"
report="$(submit "${common[@]}" --job-name=tst_report \
  --dependency="afterok:${profile}" tokenshift_training/slurm/report.sbatch)"

echo "TokenShift training: smoke=${smoke} train=${train} verify=${verify} eval=${evaluate} arm_report=${arm_report} profile=${profile} report=${report}"
echo 'Twelve training runs: frozen and dual-LoRA parents x block 8/block 6 x seeds 42/43/44.'
echo 'All dependency stages were queued by this one launcher. Flickr test remains sealed.'
echo 'Codex did not submit jobs.'
