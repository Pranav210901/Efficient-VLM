#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

PHASE="all"
MODE="parallel"
PIPELINE="configs/alignment_v3_recovery/pipeline.yaml"
MAX_GPUS=8
PARTITION="teaching"
DRY_RUN=false
RESUME_REQUESTED=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME_REQUESTED=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${PHASE}" == "all" || "${PHASE}" == "screen" || "${PHASE}" == "recipes" || "${PHASE}" == "final" ]] || {
  echo "--phase must be all, screen, recipes, or final" >&2
  exit 2
}
[[ "${MODE}" == "parallel" || "${MODE}" == "sequential" ]] || {
  echo "--mode must be parallel or sequential" >&2
  exit 2
}
[[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-total-gpus must be positive" >&2
  exit 2
}

export ALIGNMENT_V3_ROOT="${ROOT}"
export ALIGNMENT_V3_PIPELINE="${PIPELINE}"
mkdir -p logs/alignment_v3_recovery/slurm results/alignment_v3_recovery

submit() {
  local dry_id="$1"
  shift
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "${dry_id}"
  else
    timeout --foreground "${ALIGNMENT_V3_SBATCH_TIMEOUT_SECONDS:-60}" sbatch --parsable "$@"
  fi
}

cpu() {
  local dry="$1" name="$2" command="$3" dependency="$4"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,ALIGNMENT_V3_ROOT=${ROOT},ALIGNMENT_V3_PIPELINE=${PIPELINE},ALIGNMENT_V3_COMMAND=${command}" slurm/alignment_v3_recovery/cpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

gpu() {
  local dry="$1" name="$2" command="$3" dependency="$4" count="$5" limit="$6"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  if (( count > 1 )); then args+=(--array="0-$((count - 1))%${limit}"); fi
  args+=(--export="ALL,ALIGNMENT_V3_ROOT=${ROOT},ALIGNMENT_V3_PIPELINE=${PIPELINE},ALIGNMENT_V3_COMMAND=${command}" slurm/alignment_v3_recovery/gpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

write_submission() {
  local phase="$1"
  shift
  local destination="results/alignment_v3_recovery/slurm_submission_${phase}.json"
  "${ALIGNMENT_V3_PYTHON}" - "${destination}" "${phase}" "${MODE}" "${MAX_GPUS}" "${PARTITION}" "${DRY_RUN}" "${RESUME_REQUESTED}" "$@" <<'PY'
import json
import sys
from pathlib import Path

destination, phase, mode, max_gpus, partition, dry_run, resume_requested, *pairs = sys.argv[1:]
jobs = dict(value.split("=", 1) for value in pairs)
payload = {
    "phase": phase,
    "mode": mode,
    "max_total_gpus": int(max_gpus),
    "partition": partition,
    "dry_run": dry_run == "true",
    "resume_requested": resume_requested == "true",
    "resume_policy": "fingerprint-skip-complete; epoch-resume-from-latest",
    "jobs": jobs,
}
path = Path(destination)
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Submission manifest: {path}")
PY
}

limit="${MAX_GPUS}"
[[ "${MODE}" == "sequential" ]] && limit=1

if [[ "${PHASE}" == "all" || "${PHASE}" == "screen" ]]; then
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
  reference_count="$(( $(wc -l < results/alignment_v3_recovery/manifests/reference_jobs.csv) - 1 ))"
  pair_count="$(( $(wc -l < results/alignment_v3_recovery/manifests/pair_jobs.csv) - 1 ))"
  confirmation_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.utils.config import load_config
import os
p = load_config(os.environ["ALIGNMENT_V3_PIPELINE"])
print(int(p["recovery"]["confirmation_top_k"]) * len(p["recovery"]["confirmation_seeds"]))
PY
)"
  validate_id="$(cpu 940001 a3r_validate validate "")"
  prefetch_id="$(cpu 940002 a3r_prefetch prefetch "afterok:${validate_id}")"
  reference_id="$(gpu 940003 a3r_reference reference "afterok:${prefetch_id}" "${reference_count}" "${limit}")"
  train_id="$(gpu 940004 a3r_batch_train train-pair "afterok:${reference_id}" "${pair_count}" "${limit}")"
  eval_id="$(gpu 940005 a3r_batch_eval eval-pair "afterok:${train_id}" "${pair_count}" "${limit}")"
  shortlist_id="$(cpu 940006 a3r_shortlist select-recovery-batches "afterok:${eval_id}")"
  confirm_train_id="$(gpu 940007 a3r_confirm_train train-confirmation "afterok:${shortlist_id}" "${confirmation_count}" "${limit}")"
  confirm_eval_id="$(gpu 940008 a3r_confirm_eval eval-confirmation "afterok:${confirm_train_id}" "${confirmation_count}" "${limit}")"
  select_id="$(cpu 940009 a3r_pair_gate select-recovery-pair "afterok:${confirm_eval_id}")"
  if [[ "${PHASE}" == "all" ]]; then
    ablation_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.alignment_v3.runner import ablation_jobs
from src.utils.config import load_config
import os
print(len(ablation_jobs(load_config(os.environ["ALIGNMENT_V3_PIPELINE"]))))
PY
)"
    final_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.alignment_v3.runner import final_jobs
from src.utils.config import load_config
import os
print(len(final_jobs(load_config(os.environ["ALIGNMENT_V3_PIPELINE"]))))
PY
)"
    cache_id="$(gpu 940010 a3r_teacher_cache teacher-cache "afterok:${select_id}" 1 1)"
    recipe_train_id="$(gpu 940011 a3r_recipe_train train-ablation "afterok:${cache_id}" "${ablation_count}" "${limit}")"
    recipe_eval_id="$(gpu 940012 a3r_recipe_eval eval-ablation "afterok:${recipe_train_id}" "${ablation_count}" "${limit}")"
    recipe_select_id="$(cpu 940013 a3r_recipe_gate select-recovery-recipe "afterok:${recipe_eval_id}")"
    final_train_id="$(gpu 940014 a3r_final_train train-final "afterok:${recipe_select_id}" "${final_count}" "${limit}")"
    final_eval_id="$(gpu 940015 a3r_final_eval eval-final "afterok:${final_train_id}" "${final_count}" "${limit}")"
    oracle_id="$(cpu 940016 a3r_oracle oracle "afterok:${final_eval_id}")"
    report_id="$(cpu 940017 a3r_report report "afterok:${oracle_id}")"
    write_submission all \
      "validate=${validate_id}" "prefetch=${prefetch_id}" "reference=${reference_id}" \
      "batch_train=${train_id}" "batch_eval=${eval_id}" \
      "shortlist=${shortlist_id}" "confirmation_train=${confirm_train_id}" \
      "confirmation_eval=${confirm_eval_id}" "pair_gate=${select_id}" \
      "teacher_cache=${cache_id}" "recipe_train=${recipe_train_id}" \
      "recipe_eval=${recipe_eval_id}" "recipe_gate=${recipe_select_id}" \
      "final_train=${final_train_id}" "final_eval=${final_eval_id}" \
      "oracle=${oracle_id}" "report=${report_id}"
  else
    write_submission screen \
      "validate=${validate_id}" "prefetch=${prefetch_id}" "reference=${reference_id}" \
      "batch_train=${train_id}" "batch_eval=${eval_id}" \
      "shortlist=${shortlist_id}" "confirmation_train=${confirm_train_id}" \
      "confirmation_eval=${confirm_eval_id}" "pair_gate=${select_id}"
  fi
elif [[ "${PHASE}" == "recipes" ]]; then
  "${ALIGNMENT_V3_PYTHON}" - <<'PY'
import json
from pathlib import Path
p = Path("results/alignment_v3_recovery/selection/pair.json")
if not p.is_file() or not json.loads(p.read_text()).get("proceed"):
    raise SystemExit("Recovery pair gate has not passed; recipe phase was not submitted.")
PY
  reference_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.utils.config import load_config
import os
print(len(load_config(os.environ["ALIGNMENT_V3_PIPELINE"])["references"]))
PY
)"
  ablation_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.alignment_v3.runner import ablation_jobs
from src.utils.config import load_config
import os
print(len(ablation_jobs(load_config(os.environ["ALIGNMENT_V3_PIPELINE"]))))
PY
)"
  reference_id="$(gpu 941001 a3r_reference reference "" "${reference_count}" "${limit}")"
  cache_id="$(gpu 941002 a3r_teacher_cache teacher-cache "" 1 1)"
  train_id="$(gpu 941003 a3r_recipe_train train-ablation "afterok:${cache_id}" "${ablation_count}" "${limit}")"
  eval_id="$(gpu 941004 a3r_recipe_eval eval-ablation "afterok:${train_id}" "${ablation_count}" "${limit}")"
  select_id="$(cpu 941005 a3r_recipe_gate select-recovery-recipe "afterok:${eval_id}:${reference_id}")"
  write_submission recipes \
    "reference=${reference_id}" "teacher_cache=${cache_id}" \
    "recipe_train=${train_id}" "recipe_eval=${eval_id}" "recipe_gate=${select_id}"
else
  "${ALIGNMENT_V3_PYTHON}" - <<'PY'
import json
from pathlib import Path
p = Path("results/alignment_v3_recovery/selection/recipe.json")
payload = json.loads(p.read_text()) if p.is_file() else {}
if not str(payload.get("status", "")).startswith("SELECTED"):
    raise SystemExit("Recovery recipe gate has not passed; final phase was not submitted.")
PY
  final_count="$("${ALIGNMENT_V3_PYTHON}" - <<'PY'
from src.alignment_v3.runner import final_jobs
from src.utils.config import load_config
import os
print(len(final_jobs(load_config(os.environ["ALIGNMENT_V3_PIPELINE"]))))
PY
)"
  train_id="$(gpu 942001 a3r_final_train train-final "" "${final_count}" "${limit}")"
  eval_id="$(gpu 942002 a3r_final_eval eval-final "afterok:${train_id}" "${final_count}" "${limit}")"
  oracle_id="$(cpu 942003 a3r_oracle oracle "afterok:${eval_id}")"
  report_id="$(cpu 942004 a3r_report report "afterok:${oracle_id}")"
  write_submission final \
    "final_train=${train_id}" "final_eval=${eval_id}" \
    "oracle=${oracle_id}" "report=${report_id}"
fi

echo "Alignment v3 recovery submitted: phase=${PHASE} mode=${MODE} max_total_gpus=${MAX_GPUS}"
