#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

mkdir -p \
  logs/gap_closure \
  logs/queue_mechanism/slurm \
  results/queue_mechanism/{manifests,runs,report} \
  artifacts/07_final_evaluation/compositional/{logs,manifests,per_run,report}

manifest="artifacts/07_final_evaluation/compositional/manifests/data_manifest.json"
if [[ ! -f "${manifest}" ]]; then
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final acquire-sugarcrepe
  if [[ ! -f data/multitask/winoground/data/test-00000-of-00001.parquet ]]; then
    echo "Winoground is not licensed locally; freezing the preregistered SugarCrepe primary only."
  fi
  "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final freeze
fi

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.compositional_final validate
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism validate

compositional="$(sbatch --parsable --array=0-8 slurm/compositional_final/evaluate.sbatch)"
compositional_report="$(
  sbatch --parsable --dependency="afterok:${compositional}" slurm/compositional_final/report.sbatch
)"

capacity="$(sbatch --parsable --array=0-5 slurm/queue_mechanism/train.sbatch)"
mechanism="$(
  sbatch --parsable --dependency="afterok:${capacity}" --array=6-14 slurm/queue_mechanism/train.sbatch
)"
queue_report="$(
  sbatch --parsable --dependency="afterany:${mechanism}" slurm/queue_mechanism/report.sbatch
)"

finalize="$(
  sbatch --parsable \
    --dependency="afterany:${compositional_report}:${queue_report}" \
    slurm/gap_closure/finalize.sbatch
)"

echo "Submitted the complete SugarCrepe gap-closure graph."
echo "Compositional evaluation: ${compositional}"
echo "Compositional report:     ${compositional_report}"
echo "Queue capacity:           ${capacity}"
echo "Queue mechanisms:         ${mechanism}"
echo "Queue report:             ${queue_report}"
echo "Final verification:       ${finalize}"
echo "Monitor with: squeue -u ${USER}"
