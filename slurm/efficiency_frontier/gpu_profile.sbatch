#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=teaching
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/efficiency_frontier/slurm/%x_%A_%a.out
#SBATCH --error=logs/efficiency_frontier/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/efficiency_frontier/common.sh
run_efficiency_frontier_stage
