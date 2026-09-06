#!/usr/bin/env bash
#SBATCH --job-name=a2_validate
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v2/00_validate/%j.out
#SBATCH --error=logs/alignment_v2/00_validate/%j.err
set -euo pipefail
source slurm/alignment_v2/common.sh
alignment_v2_diagnostics
run_alignment_v2_stage 00_validate "${ALIGNMENT_V2_PYTHON}" -m src.alignment_v2.runner validate --pipeline "${ALIGNMENT_V2_PIPELINE}"
run_alignment_v2_notebook_stage validate
