#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PARTITION=teaching
MAX_GPUS=8
DRY_RUN=false
SKIP_DOWNLOAD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus) MAX_GPUS="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$PARTITION" == teaching ]] || {
  echo "Final comparison is restricted to teaching RTX PRO 6000 Blackwell." >&2
  exit 2
}
[[ "$MAX_GPUS" =~ ^[1-8]$ ]] || {
  echo "--max-total-gpus must be between 1 and 8." >&2
  exit 2
}

mkdir -p artifacts/07_final_evaluation/zero_shot/{logs,manifests,per_run,report}
echo "Validating the frozen nine-job roster and six checkpoint fingerprints..." >&2
scripts/final_zero_shot_python.sh -m src.alignment_v3.final_zero_shot validate

if [[ "$SKIP_DOWNLOAD" == true ]]; then
  scripts/final_zero_shot_python.sh -m src.alignment_v3.final_zero_shot prepare
else
  echo "Preparing missing public zero-shot datasets on the submit node..." >&2
  scripts/final_zero_shot_python.sh -m src.alignment_v3.final_zero_shot prepare --download
fi

submit() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "99${RANDOM}"
  else
    timeout --foreground 60 sbatch --parsable "$@"
  fi
}

export_spec="ALL,FINAL_ZERO_SHOT_ROOT=${ROOT}"
evaluation="$(submit \
  --partition="$PARTITION" \
  --job-name=fzs_eval \
  --array="0-8%${MAX_GPUS}" \
  --export="$export_spec" \
  slurm/final_zero_shot/evaluate.sbatch)"
report="$(submit \
  --partition="$PARTITION" \
  --job-name=fzs_report \
  --dependency="afterok:${evaluation}" \
  --export="$export_spec" \
  slurm/final_zero_shot/report.sbatch)"

echo "Final zero-shot graph: evaluation=${evaluation} report=${report}"
echo "Nine independent one-GPU evaluation tasks (six student seeds, three references); concurrency=${MAX_GPUS}."
echo "No training jobs are included."
