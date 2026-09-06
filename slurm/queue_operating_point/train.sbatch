#!/usr/bin/env bash
# Queue decomposition at the M-T1 operating point.
#
# 15 cells: 3 seeds x { none } (age-independent, trained once per seed)
#         + 3 seeds x { both, both_fresh } x { age 16, age 64 }
#
#   sbatch --array=0-14 slurm/queue_operating_point/train.sbatch
#
# Every arm is trained here: unlike the Q04 identification control, no matched
# stale or queue-free arm exists at this operating point to reuse.  Each task
# trains one cell then evaluates it.  When the array finishes:
#   python -m src.alignment_v3.queue_operating_point report
#
# BEFORE SUBMITTING: predictions/queue_operating_point.json is a DRAFT.  Review
# and edit it, set status to PREREGISTERED, and only then submit.  The training
# entry point calls prediction_gate() but the gate only checks that statements
# are non-empty; it cannot tell a reviewed prediction from an unreviewed one.
#SBATCH --job-name=queue_oppoint
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/queue_operating_point/slurm/%x_%A_%a.out
#SBATCH --error=logs/queue_operating_point/slurm/%x_%A_%a.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
index="${SLURM_ARRAY_TASK_ID:?}"

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_operating_point train --index "${index}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_operating_point eval  --index "${index}"
