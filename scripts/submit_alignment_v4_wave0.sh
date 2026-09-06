#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/alignment_v4_wave0/pipeline.yaml"
PARTITION="teaching"
MAX_GPUS=8
MODE="parallel"
PHASE="all"
DRY_RUN=false
RESUME_REQUESTED=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME_REQUESTED=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${MODE}" == "parallel" || "${MODE}" == "sequential" ]] || {
  echo "--mode must be parallel or sequential" >&2; exit 2;
}
[[ "${PHASE}" == "all" || "${PHASE}" == "smoke" ]] || {
  echo "--phase must be all or smoke" >&2; exit 2;
}
[[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-total-gpus must be positive" >&2; exit 2;
}
limit="${MAX_GPUS}"
[[ "${MODE}" == "sequential" ]] && limit=1

# This check occurs before any ledger write, archive, or sbatch submission.
"${ALIGNMENT_V3_PYTHON}" - "${PIPELINE}" <<'PY'
import json, sys
from pathlib import Path
from src.utils.config import load_config
p = load_config(sys.argv[1])
path = Path(p["wave0"]["predictions_path"])
if not path.is_file():
    raise SystemExit(f"BLOCKED: missing prediction file {path}")
payload = json.loads(path.read_text())
expected = {"replication_anchor", "wave0_winner", "batch_effect"}
values = payload.get("predictions", [])
ids = {str(value.get("id")) for value in values}
if ids != expected:
    raise SystemExit(f"BLOCKED: prediction ids {sorted(ids)} != {sorted(expected)}")
if any(not str(value.get("statement", "")).strip() for value in values):
    raise SystemExit("BLOCKED: every prediction statement must be non-empty")
print(f"Prediction gate: PASS ({path})")
PY

mkdir -p logs/alignment_v4_wave0/slurm results/alignment_v4_wave0
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY-RUN would create results/alignment_v4_wave0/code_snapshot.tar.gz from src/ and configs/" >&2
else
  tar czf results/alignment_v4_wave0/code_snapshot.tar.gz src/ configs/
fi

submit() {
  local dry_id="$1"; shift
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "${dry_id}"
  else
    timeout --foreground "${ALIGNMENT_WAVE0_SBATCH_TIMEOUT_SECONDS:-60}" \
      sbatch --parsable "$@"
  fi
}
cpu() {
  local dry="$1" name="$2" command="$3" dependency="$4"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=${command}" slurm/alignment_v4_wave0/cpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}
gpu() {
  local dry="$1" name="$2" command="$3" dependency="$4" count="$5"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  (( count <= 1 )) || args+=(--array="0-$((count - 1))%${limit}")
  args+=(--export="ALL,ALIGNMENT_WAVE0_ROOT=${ROOT},ALIGNMENT_WAVE0_PIPELINE=${PIPELINE},ALIGNMENT_WAVE0_COMMAND=${command}" slurm/alignment_v4_wave0/gpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}
pending() {
  local stage="$1" count="$2"
  [[ "${DRY_RUN}" == "true" ]] && {
    echo "DRY-RUN prediction gate precedes ${count} PENDING ledger checks for ${stage}" >&2
    return
  }
  for ((index=0; index<count; index++)); do
    "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner ledger-pending \
      --pipeline "${PIPELINE}" --ledger-stage "${stage}" --index "${index}"
  done
}

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
validate_id="$(cpu 960001 a4w0_validate validate "")"
prefetch_id="$(cpu 960002 a4w0_prefetch prefetch "afterok:${validate_id}")"
oom_id="$(gpu 960003 a4w0_oom oom-smoke-test "afterok:${prefetch_id}" 6)"
oom_summary_id="$(cpu 960004 a4w0_oom_summary summarize-oom-smoke "afterok:${oom_id}")"
if [[ "${PHASE}" == "smoke" ]]; then
  echo "Wave 0 OOM-smoke-only graph prepared; no LR sweep or full training was submitted."
  exit 0
fi
pending wave0-lr-calibration 9
lr_train_id="$(gpu 960005 a4w0_lr_train train-wave0-lr-calibration "afterok:${oom_summary_id}" 9)"
lr_eval_id="$(gpu 960006 a4w0_lr_eval eval-wave0-lr-calibration "afterok:${lr_train_id}" 9)"
lr_select_id="$(cpu 960007 a4w0_lr_select select-wave0-lrsweep "afterok:${lr_eval_id}")"
screen_ledger_id="$(cpu 960008 a4w0_screen_ledger ledger-wave0-screen "afterok:${lr_select_id}")"
screen_train_id="$(gpu 960009 a4w0_screen_train train-wave0-screen "afterok:${screen_ledger_id}" 18)"
screen_eval_id="$(gpu 960010 a4w0_screen_eval eval-wave0-screen "afterok:${screen_train_id}" 18)"
finalists_id="$(cpu 960011 a4w0_finalists select-wave0-finalists "afterok:${screen_eval_id}")"
select_train_id="$(gpu 960012 a4w0_select_train train-wave0-select "afterok:${finalists_id}" 6)"
select_eval_id="$(gpu 960013 a4w0_select_eval eval-wave0-select "afterok:${select_train_id}" 6)"
lock_id="$(cpu 960014 a4w0_lock select-wave0-recipe "afterok:${select_eval_id}")"
report_id="$(cpu 960015 a4w0_report report "afterok:${lock_id}")"

"${ALIGNMENT_V3_PYTHON}" - results/alignment_v4_wave0/slurm_submission.json <<PY
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
  "version": "alignment-v4-wave0-1.0",
  "dry_run": "${DRY_RUN}" == "true",
  "resume_requested": "${RESUME_REQUESTED}" == "true",
  "mode": "${MODE}",
  "max_total_gpus": ${MAX_GPUS},
  "prediction_gate": "PASS",
  "code_snapshot": "would-create" if "${DRY_RUN}" == "true" else "created",
  "jobs": {
    "validate": "${validate_id}", "prefetch": "${prefetch_id}",
    "oom_smoke": "${oom_id}", "oom_summary": "${oom_summary_id}",
    "lr_train": "${lr_train_id}",
    "lr_eval": "${lr_eval_id}", "lr_select": "${lr_select_id}",
    "screen_ledger": "${screen_ledger_id}",
    "screen_train": "${screen_train_id}", "screen_eval": "${screen_eval_id}",
    "finalists": "${finalists_id}", "select_train": "${select_train_id}",
    "select_eval": "${select_eval_id}", "lock": "${lock_id}",
    "report": "${report_id}"
  }
}
path.write_text(json.dumps(payload, indent=2) + "\\n")
print(path)
PY
echo "Wave 0 graph prepared: mode=${MODE} max_total_gpus=${MAX_GPUS} dry_run=${DRY_RUN}"
