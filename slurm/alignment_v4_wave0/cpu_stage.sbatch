#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v4_wave0/slurm/%x_%A_%a.out
#SBATCH --error=logs/alignment_v4_wave0/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/alignment_v4_wave0/common.sh
run_alignment_wave0_stage
