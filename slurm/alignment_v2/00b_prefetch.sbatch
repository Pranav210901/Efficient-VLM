#!/usr/bin/env bash
#SBATCH --job-name=a2_prefetch
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v2/00b_prefetch/%j.out
#SBATCH --error=logs/alignment_v2/00b_prefetch/%j.err
set -euo pipefail
source slurm/alignment_v2/common.sh
alignment_v2_diagnostics
run_alignment_v2_stage 00b_prefetch "${ALIGNMENT_V2_PYTHON}" -m src.alignment_v2.runner prefetch --pipeline "${ALIGNMENT_V2_PIPELINE}"
run_alignment_v2_notebook_stage prefetch
