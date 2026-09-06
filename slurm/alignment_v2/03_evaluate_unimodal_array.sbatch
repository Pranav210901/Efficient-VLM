#!/usr/bin/env bash
#SBATCH --job-name=a2_evaluate
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v2/03_evaluate_unimodal_array/%A_%a.out
#SBATCH --error=logs/alignment_v2/03_evaluate_unimodal_array/%A_%a.err
set -euo pipefail
source slurm/alignment_v2/common.sh
alignment_v2_diagnostics
run_alignment_v2_stage 03_evaluate_unimodal_array "${ALIGNMENT_V2_PYTHON}" -m src.alignment_v2.runner evaluate --pipeline "${ALIGNMENT_V2_PIPELINE}"
run_alignment_v2_notebook_stage evaluate
