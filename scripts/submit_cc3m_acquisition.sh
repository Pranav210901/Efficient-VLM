#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
PIPELINE="configs/cc3m_scale/pipeline.yaml"

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_acquisition \
  retention-validate --pipeline "${PIPELINE}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.cc3m_acquisition \
  mirror-preflight --pipeline "${PIPELINE}" >/dev/null

read -r shards concurrency weeks interval < <(
  "${ALIGNMENT_V3_PYTHON}" - "${PIPELINE}" <<'PY'
import sys, yaml
p=yaml.safe_load(open(sys.argv[1]))
print(p["acquisition"]["download_train_shards"],
      p["resources"]["acquisition"]["array_concurrency"],
      p["retention"]["scheduled_weeks"],
      p["retention"]["interval_days"])
PY
)

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY-RUN: schedule ${weeks} independent weekly retention jobs."
  echo "DRY-RUN: initial retention -> ${shards}-task download array (%${concurrency}) -> ${shards}-task inspect array -> finalise."
  echo "DRY-RUN: no training and no feature-cache work."
  exit 0
fi

mkdir -p logs/cc3m_scale/{slurm,retention} data/cc3m_v1_1/{shards/train,ledgers,manifests}
schedule_manifest="data/cc3m_v1_1/manifests/retention_schedule.tsv"
temporary="${schedule_manifest}.tmp"
printf 'week\\tbegin\\tjob_id\\n' > "${temporary}"
initial_touch_id=""
for ((week=0; week<weeks; week++)); do
  begin="$(date -d "+$((week * interval)) days" '+%Y-%m-%dT%H:%M:%S')"
  job_id="$(timeout --foreground 60 sbatch --parsable \
    --begin="${begin}" \
    --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE}" \
    slurm/cc3m_scale/retention.sbatch)"
  printf '%s\\t%s\\t%s\\n' "${week}" "${begin}" "${job_id}" >> "${temporary}"
  if [[ "${week}" -eq 0 ]]; then initial_touch_id="${job_id}"; fi
done
mv "${temporary}" "${schedule_manifest}"

download_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${initial_touch_id}" --array="0-$((shards - 1))%${concurrency}" \
  --job-name=cc3m_download \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE},CC3M_COMMAND=download-shard" \
  slurm/cc3m_scale/acquisition.sbatch)"
inspect_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${download_id}" --array="0-$((shards - 1))%${concurrency}" \
  --job-name=cc3m_inspect \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE},CC3M_COMMAND=inspect-shard" \
  slurm/cc3m_scale/acquisition.sbatch)"
finalise_id="$(timeout --foreground 60 sbatch --parsable \
  --dependency="afterok:${inspect_id}" --job-name=cc3m_finalise \
  --export="ALL,CC3M_ROOT=${ROOT},CC3M_PIPELINE=${PIPELINE},CC3M_COMMAND=finalise" \
  slurm/cc3m_scale/finalise.sbatch)"
echo "CC3M retention schedule: ${schedule_manifest}"
echo "CC3M acquisition: initial_touch=${initial_touch_id} download=${download_id} inspect=${inspect_id} finalise=${finalise_id}"
