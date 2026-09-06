#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/teacher_extension_224/slurm/%x_%j.out
#SBATCH --error=logs/teacher_extension_224/slurm/%x_%j.err
set -euo pipefail

cd "${TEACHER_EXTENSION_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
export ALIGNMENT_WAVE0_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-12}"
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.runner teacher-cache \
  --pipeline configs/teacher_extension_224/pipeline_openclip.yaml

