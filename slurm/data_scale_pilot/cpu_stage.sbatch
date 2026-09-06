#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/data_scale_pilot/slurm/%x_%j.out
#SBATCH --error=logs/data_scale_pilot/slurm/%x_%j.err
set -euo pipefail
source slurm/data_scale_pilot/common.sh
