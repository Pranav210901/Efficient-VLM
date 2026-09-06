#!/usr/bin/env bash
#SBATCH --job-name=p2_report
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/10_build_phase2_report/%j.out
#SBATCH --error=logs/phase2/10_build_phase2_report/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/reporting.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 10_build_phase2_report "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.build_report --config "$CONFIG"
