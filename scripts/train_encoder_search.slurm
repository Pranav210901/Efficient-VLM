#!/bin/bash
#SBATCH --job-name=vlm_encoder_search
#SBATCH --partition=3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=fs_weka

set -euo pipefail

cd "$(dirname "$0")/.."

module load miniconda || module load anaconda
conda activate vlm

export HF_HOME=/mnt/fast/nobackup/users/pp01184/vlm_project/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/fast/nobackup/users/pp01184/vlm_project/cache/huggingface
export TORCH_HOME=/mnt/fast/nobackup/users/pp01184/vlm_project/cache/torch

mkdir -p logs checkpoints results
mkdir -p /mnt/fast/nobackup/users/pp01184/vlm_project/cache

python scripts/run_encoder_matrix.py \
  --config configs/experiment_blf.yaml \
  --mode all \
  --results_path results/alignment_matrix_results.csv
