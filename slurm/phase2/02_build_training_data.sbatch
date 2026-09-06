#!/usr/bin/env bash
#SBATCH --job-name=p2_data
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/02_build_training_data/%j.out
#SBATCH --error=logs/phase2/02_build_training_data/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/data_build.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 02_build_training_data "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.build_datasets --config "$CONFIG"
