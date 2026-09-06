from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool


def initialise_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device, distributed)


def seed_rank(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sampler_for(dataset, context: DistributedContext, shuffle: bool = True):
    return DistributedSampler(dataset, context.world_size, context.rank, shuffle=shuffle) if context.distributed else None


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= context.world_size
    return value


def rank_zero(context: DistributedContext) -> bool:
    return context.rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
