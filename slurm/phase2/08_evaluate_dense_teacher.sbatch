#!/usr/bin/env bash
#SBATCH --job-name=p2_dense_eval
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/08_evaluate_dense_teacher/%j.out
#SBATCH --error=logs/phase2/08_evaluate_dense_teacher/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/dense_teacher.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 08_evaluate_dense_teacher "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.evaluate_dense_teacher --config "$CONFIG"
