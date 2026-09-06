#!/usr/bin/env bash
#SBATCH --job-name=p2_smoke
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/03_smoke_cross_attention/%j.out
#SBATCH --error=logs/phase2/03_smoke_cross_attention/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/smoke.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 03_smoke_cross_attention "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.smoke_test --config "$CONFIG"
