#!/usr/bin/env bash
#SBATCH --job-name=p2_bridges
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%8
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/04_train_bridges_array/%A_%a.out
#SBATCH --error=logs/phase2/04_train_bridges_array/%A_%a.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/bridge_training.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 04_train_bridges_array "$CONFIG" "$PHASE2_PYTHON" -m src.phase2.cli.train_bridge --config "$CONFIG" --resume
