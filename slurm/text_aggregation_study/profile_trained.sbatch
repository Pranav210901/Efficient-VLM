#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/text_aggregation_study/slurm/%x_%j.out
#SBATCH --error=logs/text_aggregation_study/slurm/%x_%j.err
set -euo pipefail
cd "${TEXT_AGG_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
case "${TEXT_AGG_BRANCH:?}" in
  minilm_aggregation) cell="$(${ALIGNMENT_V3_PYTHON} -m src.alignment_v3.text_aggregation_study verify-gate --encoder minilm --purpose aggregation | ${ALIGNMENT_V3_PYTHON} -c 'import json,sys; print(json.load(sys.stdin)["selected_cell"])')"; [[ "$cell" == M_T1 ]] && pipeline=configs/text_aggregation_study/pipeline_m_t1.yaml || pipeline=configs/text_aggregation_study/pipeline_m_t1_small.yaml ;;
  e5_baseline) pipeline=configs/text_aggregation_study/pipeline_e_t0.yaml ;;
  e5_aggregation) cell="$(${ALIGNMENT_V3_PYTHON} -m src.alignment_v3.text_aggregation_study verify-gate --encoder e5 --purpose aggregation | ${ALIGNMENT_V3_PYTHON} -c 'import json,sys; print(json.load(sys.stdin)["selected_cell"])')"; [[ "$cell" == E_T1 ]] && pipeline=configs/text_aggregation_study/pipeline_e_t1.yaml || pipeline=configs/text_aggregation_study/pipeline_e_t1_small.yaml ;;
esac
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.resolution_distillation_trajectory profile --pipeline "$pipeline"
