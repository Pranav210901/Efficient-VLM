#!/usr/bin/env bash

# Configure Slurm and runtime resources for one Wave 0 submission graph.
# This file is sourced by submitters and deliberately does not submit jobs.
alignment_v4_wave0_partition_profile() {
  local requested_partition="$1"
  local requested_max_gpus="${2:-}"

  case "${requested_partition}" in
    teaching)
      ALIGNMENT_WAVE0_PARTITION="teaching"
      ALIGNMENT_WAVE0_PARTITION_GPU_LIMIT=8
      ALIGNMENT_WAVE0_NODE_CPU_LIMIT=92
      ALIGNMENT_WAVE0_TARGET_CPUS_PER_GPU=12
      ALIGNMENT_WAVE0_GPU_MEM="48G"
      ALIGNMENT_WAVE0_CPU_CPUS=2
      ALIGNMENT_WAVE0_CPU_MEM="16G"
      ALIGNMENT_WAVE0_NUM_WORKERS=10
      ALIGNMENT_WAVE0_PRECISION="bf16"
      ALIGNMENT_WAVE0_GPU_SBATCH="slurm/alignment_v4_wave0/gpu_stage.sbatch"
      ALIGNMENT_WAVE0_CPU_SBATCH="slurm/alignment_v4_wave0/cpu_stage.sbatch"
      ;;
    2080ti)
      ALIGNMENT_WAVE0_PARTITION="2080ti"
      ALIGNMENT_WAVE0_PARTITION_GPU_LIMIT=16
      # One process plus four data-loader workers.
      ALIGNMENT_WAVE0_GPU_CPUS=5
      ALIGNMENT_WAVE0_GPU_MEM="32G"
      ALIGNMENT_WAVE0_CPU_CPUS=2
      ALIGNMENT_WAVE0_CPU_MEM="8G"
      ALIGNMENT_WAVE0_NUM_WORKERS=4
      # AISurrey's PyTorch 2.8 stack exposes emulated BF16 on Turing.
      ALIGNMENT_WAVE0_PRECISION="bf16"
      # The teaching templates require fs_weka, which is not a valid node
      # constraint on the 2080ti partition.
      ALIGNMENT_WAVE0_GPU_SBATCH="slurm/alignment_v4_wave0/gpu_stage_2080ti.sbatch"
      ALIGNMENT_WAVE0_CPU_SBATCH="slurm/alignment_v4_wave0/cpu_stage_2080ti.sbatch"
      ;;
    *)
      echo "--partition must be teaching or 2080ti" >&2
      return 2
      ;;
  esac

  if [[ -z "${requested_max_gpus}" ]]; then
    ALIGNMENT_WAVE0_MAX_GPUS="${ALIGNMENT_WAVE0_PARTITION_GPU_LIMIT}"
  elif [[ "${requested_max_gpus}" =~ ^[1-9][0-9]*$ ]]; then
    ALIGNMENT_WAVE0_MAX_GPUS="${requested_max_gpus}"
  else
    echo "--max-total-gpus must be a positive integer" >&2
    return 2
  fi
  if (( ALIGNMENT_WAVE0_MAX_GPUS > ALIGNMENT_WAVE0_PARTITION_GPU_LIMIT )); then
    echo "Requested ${ALIGNMENT_WAVE0_MAX_GPUS} GPUs, but ${ALIGNMENT_WAVE0_PARTITION} is capped at ${ALIGNMENT_WAVE0_PARTITION_GPU_LIMIT}." >&2
    return 2
  fi

  if [[ "${ALIGNMENT_WAVE0_PARTITION}" == "teaching" ]]; then
    ALIGNMENT_WAVE0_GPU_CPUS=$(
      echo $((ALIGNMENT_WAVE0_NODE_CPU_LIMIT / ALIGNMENT_WAVE0_MAX_GPUS))
    )
    if (( ALIGNMENT_WAVE0_GPU_CPUS > ALIGNMENT_WAVE0_TARGET_CPUS_PER_GPU )); then
      ALIGNMENT_WAVE0_GPU_CPUS="${ALIGNMENT_WAVE0_TARGET_CPUS_PER_GPU}"
    fi
  fi
  ALIGNMENT_WAVE0_TOTAL_GPU_CPUS=$(
    echo $((ALIGNMENT_WAVE0_GPU_CPUS * ALIGNMENT_WAVE0_MAX_GPUS))
  )

  export ALIGNMENT_WAVE0_PARTITION
  export ALIGNMENT_WAVE0_MAX_GPUS
  export ALIGNMENT_WAVE0_GPU_CPUS
  export ALIGNMENT_WAVE0_TOTAL_GPU_CPUS
  export ALIGNMENT_WAVE0_NUM_WORKERS
  export ALIGNMENT_WAVE0_PRECISION
  export ALIGNMENT_WAVE0_GPU_SBATCH
  export ALIGNMENT_WAVE0_CPU_SBATCH
}
