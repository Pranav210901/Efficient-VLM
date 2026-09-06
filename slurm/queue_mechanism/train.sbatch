#!/usr/bin/env bash
#SBATCH --job-name=queue_mech
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/queue_mechanism/slurm/%x_%A_%a.out
#SBATCH --error=logs/queue_mechanism/slurm/%x_%A_%a.err
set -euo pipefail
cd "${ALIGNMENT_VLM_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
index="${SLURM_ARRAY_TASK_ID:?}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism train --index "${index}"
"${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.queue_mechanism eval --index "${index}"
