#!/bin/bash
#SBATCH --job-name=blf_vlm
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

export HF_HOME=/mnt/fast/nobackup/users/pp01184/alignment_vlm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/fast/nobackup/users/pp01184/alignment_vlm/cache/huggingface
export TORCH_HOME=/mnt/fast/nobackup/users/pp01184/alignment_vlm/cache/torch

mkdir -p logs checkpoints results
mkdir -p /mnt/fast/nobackup/users/pp01184/alignment_vlm/cache

python scripts/train.py --config configs/experiment_blf.yaml
