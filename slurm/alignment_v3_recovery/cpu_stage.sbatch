#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v3_recovery/slurm/%x_%j.out
#SBATCH --error=logs/alignment_v3_recovery/slurm/%x_%j.err
set -euo pipefail
source slurm/alignment_v3_recovery/common.sh
alignment_v3_recovery_diagnostics
run_alignment_v3_recovery_stage

