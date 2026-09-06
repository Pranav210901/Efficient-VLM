#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/data_scale_pilot/slurm/%x_%A_%a.out
#SBATCH --error=logs/data_scale_pilot/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/data_scale_pilot/common.sh
