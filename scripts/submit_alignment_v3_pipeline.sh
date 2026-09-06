#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

MODE="parallel"
PIPELINE="configs/alignment_v3/pipeline.yaml"
MAX_GPUS=8
PARTITION="teaching"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) shift ;; # Fingerprint-aware resume is always enabled.
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${MODE}" == "parallel" || "${MODE}" == "sequential" ]] || { echo "Invalid --mode" >&2; exit 2; }
[[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "--max-total-gpus must be positive" >&2; exit 2; }

export ALIGNMENT_V3_ROOT="${ROOT}"
export ALIGNMENT_V3_PIPELINE="${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${PIPELINE}"
manifest_root="results/alignment_v3/manifests"
reference_count="$(( $(wc -l < "${manifest_root}/reference_jobs.csv") - 1 ))"
pair_count="$(( $(wc -l < "${manifest_root}/pair_jobs.csv") - 1 ))"
sensitivity_count="$(( $(wc -l < "${manifest_root}/sensitivity_jobs.csv") - 1 ))"
ablation_count="$(( $(wc -l < "${manifest_root}/ablation_jobs.csv") - 1 ))"
final_count="$(( $(wc -l < "${manifest_root}/final_jobs.csv") - 1 ))"
transfer_count="$(( $(wc -l < "${manifest_root}/transfer_jobs.csv") - 1 ))"
mkdir -p logs/alignment_v3/slurm

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
  args+=(--export="ALL,ALIGNMENT_V3_ROOT=${ROOT},ALIGNMENT_V3_PIPELINE=${PIPELINE},ALIGNMENT_V3_COMMAND=${command}" slurm/alignment_v3/cpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

gpu() {
  local dry="$1" name="$2" command="$3" dependency="$4" count="$5" limit="$6"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  if (( count > 1 )); then args+=(--array="0-$((count - 1))%${limit}"); fi
  args+=(--export="ALL,ALIGNMENT_V3_ROOT=${ROOT},ALIGNMENT_V3_PIPELINE=${PIPELINE},ALIGNMENT_V3_COMMAND=${command}" slurm/alignment_v3/gpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

limit="${MAX_GPUS}"
[[ "${MODE}" == "sequential" ]] && limit=1
validate_id="$(cpu 930001 a3_validate validate "")"
prefetch_id="$(cpu 930002 a3_prefetch prefetch "afterok:${validate_id}")"

if (( MAX_GPUS > 1 )) && [[ "${MODE}" == "parallel" ]]; then
  reference_limit=$((MAX_GPUS - 1))
  (( reference_limit > reference_count )) && reference_limit="${reference_count}"
  reference_id="$(gpu 930003 a3_reference reference "afterok:${prefetch_id}" "${reference_count}" "${reference_limit}")"
  probe_id="$(gpu 930004 a3_batch_probe batch-probe "afterok:${prefetch_id}" 1 1)"
else
  reference_id="$(gpu 930003 a3_reference reference "afterok:${prefetch_id}" "${reference_count}" 1)"
  probe_id="$(gpu 930004 a3_batch_probe batch-probe "afterok:${reference_id}" 1 1)"
fi

pair_train_id="$(gpu 930005 a3_pair_train train-pair "afterok:${reference_id}:${probe_id}" "${pair_count}" "${limit}")"
pair_eval_id="$(gpu 930006 a3_pair_eval eval-pair "afterok:${pair_train_id}" "${pair_count}" "${limit}")"
pair_select_id="$(cpu 930007 a3_pair_select select-pair "afterok:${pair_eval_id}")"
cache_id="$(gpu 930008 a3_teacher_cache teacher-cache "afterok:${pair_select_id}" 1 1)"
sensitivity_train_id="$(gpu 930009 a3_distill_train train-sensitivity "afterok:${cache_id}" "${sensitivity_count}" "${limit}")"
sensitivity_eval_id="$(gpu 930010 a3_distill_eval eval-sensitivity "afterok:${sensitivity_train_id}" "${sensitivity_count}" "${limit}")"
distill_select_id="$(cpu 930011 a3_distill_select select-distillation "afterok:${sensitivity_eval_id}")"
component_smoke_id="$(gpu 930012 a3_component_smoke component-smoke "afterok:${distill_select_id}" 1 1)"
ablation_train_id="$(gpu 930013 a3_ablation_train train-ablation "afterok:${component_smoke_id}" "${ablation_count}" "${limit}")"
ablation_eval_id="$(gpu 930014 a3_ablation_eval eval-ablation "afterok:${ablation_train_id}" "${ablation_count}" "${limit}")"
recipe_select_id="$(cpu 930015 a3_recipe_select select-recipe "afterok:${ablation_eval_id}")"
final_train_id="$(gpu 930016 a3_final_train train-final "afterok:${recipe_select_id}" "${final_count}" "${limit}")"
final_eval_id="$(gpu 930017 a3_final_eval eval-final "afterok:${final_train_id}" "${final_count}" "${limit}")"
oracle_id="$(cpu 930018 a3_oracle oracle "afterok:${final_eval_id}")"
flickr_path="$("${ALIGNMENT_V3_PYTHON}" - <<PY
from src.utils.config import load_config
print(load_config("${PIPELINE}")["optional_transfer"]["flickr30k_csv"])
PY
)"
if [[ -f "${flickr_path}" ]]; then
  transfer_id="$(gpu 930019 a3_transfer transfer "afterok:${final_eval_id}" "${transfer_count}" "${limit}")"
  report_dependency="afterok:${oracle_id}:${transfer_id}"
else
  transfer_id="SKIPPED_OPTIONAL_MISSING"
  report_dependency="afterok:${oracle_id}"
  echo "Alignment v3: optional Flickr30k CSV is absent (${flickr_path}); transfer stage will not be submitted." >&2
fi
report_id="$(cpu 930020 a3_report report "${report_dependency}")"

submission="results/alignment_v3/slurm_submission.json"
"${ALIGNMENT_V3_PYTHON}" - <<PY
import json
from pathlib import Path
payload = {
  "mode": "${MODE}", "partition": "${PARTITION}", "max_total_gpus": ${MAX_GPUS},
  "dry_run": "${DRY_RUN}" == "true",
  "jobs": {
    "validate": "${validate_id}", "prefetch": "${prefetch_id}", "reference": "${reference_id}",
    "batch_probe": "${probe_id}", "pair_train": "${pair_train_id}", "pair_eval": "${pair_eval_id}",
    "pair_select": "${pair_select_id}", "teacher_cache": "${cache_id}",
    "sensitivity_train": "${sensitivity_train_id}", "sensitivity_eval": "${sensitivity_eval_id}",
    "distillation_select": "${distill_select_id}", "component_smoke": "${component_smoke_id}",
    "ablation_train": "${ablation_train_id}",
    "ablation_eval": "${ablation_eval_id}", "recipe_select": "${recipe_select_id}",
    "final_train": "${final_train_id}", "final_eval": "${final_eval_id}",
    "transfer": "${transfer_id}", "oracle": "${oracle_id}", "report": "${report_id}"
  }
}
path=Path("${submission}"); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2)+"\\n")
PY
echo "Alignment v3 submitted: mode=${MODE} max_total_gpus=${MAX_GPUS}"
echo "Submission manifest: ${submission}"
