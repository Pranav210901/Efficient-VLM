#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtxpro6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/queue_factorial/slurm/%x_%A_%a.out
#SBATCH --error=logs/queue_factorial/slurm/%x_%A_%a.err
set -euo pipefail
source slurm/queue_factorial/common.sh
run_queue_factorial_stage
