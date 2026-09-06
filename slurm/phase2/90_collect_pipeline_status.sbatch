#!/usr/bin/env bash
#SBATCH --job-name=p2_status
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/90_collect_pipeline_status/%j.out
#SBATCH --error=logs/phase2/90_collect_pipeline_status/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/reporting.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 90_collect_pipeline_status "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.collect_failures
