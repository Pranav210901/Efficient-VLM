#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/alignment_v4_wave0/slurm/%x_%A_%a.out
#SBATCH --error=logs/alignment_v4_wave0/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/alignment_v4_wave0/common.sh
run_alignment_wave0_stage
