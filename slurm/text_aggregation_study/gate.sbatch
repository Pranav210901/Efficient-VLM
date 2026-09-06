#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/text_aggregation_study/slurm/%x_%j.out
#SBATCH --error=logs/text_aggregation_study/slurm/%x_%j.err
set -euo pipefail
cd "${TEXT_AGG_ROOT:?}"
source scripts/alignment_v3_env.sh
alignment_v3_resolve_python
exec "${ALIGNMENT_V3_PYTHON}" -m src.alignment_v3.text_aggregation_study gate --encoder "${TEXT_AGG_ENCODER:?}"
