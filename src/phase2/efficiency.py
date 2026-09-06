from __future__ import annotations

import time
from typing import Callable

import torch


def profile_callable(function: Callable[[], object], device: str | torch.device = "cpu", warmup: int = 2, repetitions: int = 10) -> dict[str, float]:
    target = torch.device(device)
    for _ in range(warmup): function()
    if target.type == "cuda": torch.cuda.synchronize(target); torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    for _ in range(repetitions): function()
    if target.type == "cuda": torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started
    return {
        "latency_ms": elapsed * 1000.0 / repetitions,
        "throughput_per_second": repetitions / elapsed,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0.0,
    }
