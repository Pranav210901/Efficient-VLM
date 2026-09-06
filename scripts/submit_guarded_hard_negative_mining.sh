#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/alignment_v3_env.sh
SUBMIT_PYTHON="${ROOT}/.venv-aisurrey/bin/python"
ALIGNMENT_V3_PYTHON="${ROOT}/.venv/bin/python"
ALIGNMENT_V3_SKIP_IMPORT_PROBE=1
export ALIGNMENT_V3_PYTHON ALIGNMENT_V3_SKIP_IMPORT_PROBE
[[ -x "$SUBMIT_PYTHON" ]] || { echo "Missing submit-node Python: $SUBMIT_PYTHON" >&2; exit 2; }

PARTITION=teaching
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$PARTITION" == teaching ]] || { echo "Mining is frozen to teaching RTX PRO 6000 Blackwell." >&2; exit 2; }
mkdir -p logs/guarded_hard_negative_study/slurm results/guarded_hard_negative_study/mining/embeddings
timeout --foreground 180 "$SUBMIT_PYTHON" -m src.alignment_v3.guarded_hard_negative_study validate

submit() {
  if [[ "$DRY_RUN" == true ]]; then echo "DRY-RUN sbatch $*" >&2; echo 999001; else timeout --foreground 60 sbatch --parsable "$@"; fi
}
export_spec="ALL,GHN_ROOT=${ROOT}"
encode="$(submit --partition="$PARTITION" --job-name=ghn_encode --array=0-2%3 --export="$export_spec" slurm/guarded_hard_negative_study/encode.sbatch)"
mine="$(submit --partition="$PARTITION" --job-name=ghn_mine --dependency="afterok:${encode}" --export="$export_spec" slurm/guarded_hard_negative_study/mine.sbatch)"
echo "Guarded-HN mining diagnostic: encode=${encode} mine=${mine}"
echo "This graph stops after mining feasibility; it submits no training jobs."
