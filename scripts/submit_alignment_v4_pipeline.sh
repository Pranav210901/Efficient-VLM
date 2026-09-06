#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

TEACHER="both"
MODE="parallel"
MAX_GPUS=8
PARTITION="teaching"
DRY_RUN=false
RESUME_REQUESTED=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher) TEACHER="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max-total-gpus|--max-concurrent) MAX_GPUS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME_REQUESTED=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${TEACHER}" == "both" || "${TEACHER}" == "mobileclip2" || "${TEACHER}" == "siglip2" ]] || {
  echo "--teacher must be both, mobileclip2, or siglip2" >&2; exit 2;
}
[[ "${MODE}" == "parallel" || "${MODE}" == "sequential" ]] || {
  echo "--mode must be parallel or sequential" >&2; exit 2;
}
[[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-total-gpus must be positive" >&2; exit 2;
}

mkdir -p logs/alignment_v4/slurm results/alignment_v4

submit() {
  local dry_id="$1"; shift
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY-RUN sbatch $*" >&2
    echo "${dry_id}"
  else
    timeout --foreground "${ALIGNMENT_V4_SBATCH_TIMEOUT_SECONDS:-60}" \
      sbatch --parsable "$@"
  fi
}

cpu() {
  local dry="$1" name="$2" command="$3" dependency="$4" pipeline="$5"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${pipeline},ALIGNMENT_V4_COMMAND=${command}" slurm/alignment_v4/cpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

gpu() {
  local dry="$1" name="$2" command="$3" dependency="$4" pipeline="$5" count="$6" limit="$7"
  local args=(--partition="${PARTITION}" --job-name="${name}")
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  if (( count > 1 )); then args+=(--array="0-$((count - 1))%${limit}"); fi
  args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT},ALIGNMENT_V4_PIPELINE=${pipeline},ALIGNMENT_V4_COMMAND=${command}" slurm/alignment_v4/gpu_stage.sbatch)
  submit "${dry}" "${args[@]}"
}

write_submission() {
  "${ALIGNMENT_V3_PYTHON}" - "$@" <<'PY'
import json, sys
from pathlib import Path
destination, teacher, mode, max_gpus, partition, dry, resume, *pairs = sys.argv[1:]
payload = {
    "version": "alignment-v4-capture-probe-1.0",
    "teacher": teacher,
    "mode": mode,
    "max_total_gpus": int(max_gpus),
    "partition": partition,
    "dry_run": dry == "true",
    "resume_requested": resume == "true",
    "resume_policy": "fingerprint-skip-complete; epoch-resume-from-latest",
    "jobs": dict(pair.split("=", 1) for pair in pairs),
}
path = Path(destination)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Submission manifest: {path}")
PY
}

if [[ "${TEACHER}" == "both" ]]; then
  teachers=(mobileclip2 siglip2)
else
  teachers=("${TEACHER}")
fi

limit="${MAX_GPUS}"
if [[ "${MODE}" == "sequential" ]]; then
  limit=1
elif (( ${#teachers[@]} == 2 )); then
  limit=$(( MAX_GPUS / 2 ))
  (( limit >= 1 )) || limit=1
fi

declare -a all_job_pairs=()
declare -a report_ids=()
previous_report=""
counter=0
for teacher in "${teachers[@]}"; do
  counter=$((counter + 1))
  pipeline="configs/alignment_v4/pipeline_${teacher}.yaml"
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner validate --pipeline "${pipeline}"
  count="$("${ALIGNMENT_V3_PYTHON}" - "${pipeline}" <<'PY'
import sys
from src.alignment_v3.runner import load_pipeline, sensitivity_jobs
_, pipeline = load_pipeline(sys.argv[1])
print(len(sensitivity_jobs(pipeline)))
PY
)"
  prefix="a4_mob"
  dry_base=950000
  [[ "${teacher}" == "siglip2" ]] && prefix="a4_sig" && dry_base=951000

  initial_dependency=""
  if [[ "${MODE}" == "sequential" && -n "${previous_report}" ]]; then
    initial_dependency="afterok:${previous_report}"
  fi
  validate_id="$(cpu "$((dry_base + 1))" "${prefix}_validate" validate "${initial_dependency}" "${pipeline}")"
  prefetch_id="$(cpu "$((dry_base + 2))" "${prefix}_prefetch" prefetch "afterok:${validate_id}" "${pipeline}")"
  cache_id="$(gpu "$((dry_base + 3))" "${prefix}_cache" teacher-cache "afterok:${prefetch_id}" "${pipeline}" 1 1)"
  train_id="$(gpu "$((dry_base + 4))" "${prefix}_train" train-sensitivity "afterok:${cache_id}" "${pipeline}" "${count}" "${limit}")"
  eval_id="$(gpu "$((dry_base + 5))" "${prefix}_eval" eval-sensitivity "afterok:${train_id}" "${pipeline}" "${count}" "${limit}")"
  select_id="$(cpu "$((dry_base + 6))" "${prefix}_select" select-distillation "afterok:${eval_id}" "${pipeline}")"
  report_id="$(cpu "$((dry_base + 7))" "${prefix}_report" report "afterok:${select_id}" "${pipeline}")"
  report_ids+=("${report_id}")
  previous_report="${report_id}"
  all_job_pairs+=(
    "${teacher}_validate=${validate_id}"
    "${teacher}_prefetch=${prefetch_id}"
    "${teacher}_teacher_cache=${cache_id}"
    "${teacher}_train=${train_id}"
    "${teacher}_eval=${eval_id}"
    "${teacher}_select=${select_id}"
    "${teacher}_report=${report_id}"
  )
done

if [[ "${TEACHER}" == "both" ]]; then
  combine_dependency="afterok:$(IFS=:; echo "${report_ids[*]}")"
  combine_args=(--partition="${PARTITION}" --job-name=a4_combine --dependency="${combine_dependency}")
  combine_args+=(--export="ALL,ALIGNMENT_V4_ROOT=${ROOT}" slurm/alignment_v4/combine.sbatch)
  combine_id="$(submit 952001 "${combine_args[@]}")"
  all_job_pairs+=("combined_report=${combine_id}")
fi

manifest="results/alignment_v4/slurm_submission_${TEACHER}.json"
write_submission "${manifest}" "${TEACHER}" "${MODE}" "${MAX_GPUS}" "${PARTITION}" \
  "${DRY_RUN}" "${RESUME_REQUESTED}" "${all_job_pairs[@]}"
echo "Alignment v4 submitted: teacher=${TEACHER} mode=${MODE} max_total_gpus=${MAX_GPUS}"
echo "Resume is always safe; completed fingerprint-matched jobs are skipped and interrupted training resumes from latest.pt."
