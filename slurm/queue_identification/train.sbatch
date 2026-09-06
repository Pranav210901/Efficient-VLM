#!/usr/bin/env bash
# Queue identification control: 6 cells = {age 16, 64} x {seed 42, 43, 44},
# batch 1024, queue_mode=both_fresh.  The matched stale ("both") and queue-free
# arms already exist and are NOT re-run.
#
#   sbatch --array=0-5 slurm/queue_identification/train.sbatch
#
# Resources follow configs/queue_factorial/pipeline.yaml: 12 cpus, 96G, fs_weka,
# one GPU on the teaching partition.
#
# Each task trains one cell then evaluates it.  When the array finishes:
#   python -m src.alignment_v3.queue_identification report
#SBATCH --job-name=queue_ident
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/queue_identification/slurm/%x_%A_%a.out
#SBATCH --error=logs/queue_identification/slurm/%x_%A_%a.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
index="${SLURM_ARRAY_TASK_ID:?}"

"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_identification train --index "${index}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_identification eval  --index "${index}"
