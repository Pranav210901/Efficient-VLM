#!/usr/bin/env bash
#SBATCH --job-name=p2_dense
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --time=2-23:59:00
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=fs_weka
#SBATCH --output=logs/phase2/07_train_dense_teacher/%j.out
#SBATCH --error=logs/phase2/07_train_dense_teacher/%j.err
set -euo pipefail
source slurm/phase2/common.sh
CONFIG="${PHASE2_CONFIG_ROOT:-configs/phase2}/dense_teacher.yaml"
phase2_diagnostics "$CONFIG"
run_phase2_stage 07_train_dense_teacher "$CONFIG" "$PHASE2_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" -m src.phase2.cli.train_dense_teacher --config "$CONFIG" --resume
