#!/usr/bin/env bash
#SBATCH --job-name=step16_hardneg
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=fs_weka
#SBATCH --time=16:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append
#SBATCH --chdir=/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Project: $PROJECT_ROOT"
echo "Host: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "Allocated CUDA devices: ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"
echo "Launching Step 16 on logical GPUs 0 1 2 3 with resume support"

srun \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --kill-on-bad-exit=1 \
  "$PYTHON_BIN" scripts/run_training_hard_negative_jobs.py \
    --mode full \
    --count 8 \
    --chunk-size 128 \
    --gpus 0 1 2 3 \
    --resume &

runner_pid=$!

shutdown() {
  trap - TERM INT HUP
  echo "Forwarding termination to the Step 16 coordinator..."
  kill -TERM "$runner_pid" 2>/dev/null || true
  wait "$runner_pid" || true
  exit 143
}

trap shutdown TERM INT HUP

set +e
wait "$runner_pid"
status=$?
set -e
exit "$status"
