#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/text_aggregation_study/slurm/%x_%A_%a.out
#SBATCH --error=logs/text_aggregation_study/slurm/%x_%A_%a.err
set -euo pipefail
cd "${TEXT_AGG_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
case "${TEXT_AGG_BRANCH:?}" in
  minilm_aggregation)
    cell="$(${ALIGNMENT_V3_PYTHON} -m src.alignment_v3.text_aggregation_study verify-gate --encoder minilm --purpose aggregation | ${ALIGNMENT_V3_PYTHON} -c 'import json,sys; print(json.load(sys.stdin)["selected_cell"])')"
    [[ "$cell" == M_T1 ]] && pipeline=configs/text_aggregation_study/pipeline_m_t1.yaml || pipeline=configs/text_aggregation_study/pipeline_m_t1_small.yaml ;;
  e5_baseline)
    "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.text_aggregation_study verify-gate --encoder e5 --purpose baseline >/dev/null
    pipeline=configs/text_aggregation_study/pipeline_e_t0.yaml ;;
  e5_aggregation)
    cell="$(${ALIGNMENT_V3_PYTHON} -m src.alignment_v3.text_aggregation_study verify-gate --encoder e5 --purpose aggregation | ${ALIGNMENT_V3_PYTHON} -c 'import json,sys; print(json.load(sys.stdin)["selected_cell"])')"
    [[ "$cell" == E_T1 ]] && pipeline=configs/text_aggregation_study/pipeline_e_t1.yaml || pipeline=configs/text_aggregation_study/pipeline_e_t1_small.yaml ;;
  *) echo "unknown branch ${TEXT_AGG_BRANCH}" >&2; exit 2 ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner train-sensitivity --pipeline "$pipeline"
