#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=freezeshift/logs/slurm/%x_%j.out
#SBATCH --error=freezeshift/logs/slurm/%x_%j.err
set -euo pipefail
cd "${FREEZESHIFT_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python

# Reports are regenerated sequentially in one allocation, followed by the only
# GPU-dependent stage and the final summary. No training or epoch evaluation.
for pipeline_index in 0 1 2; do
  "${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner report-arm \
    --pipeline-index "${pipeline_index}"
done
"${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner profile
"${ALIGNMENT_V3_PYTHON}" -m freezeshift.runner report
