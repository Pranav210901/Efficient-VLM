from __future__ import annotations

import os


def normalize_gpu_ids(values: list[str] | None) -> list[str]:
    """Accept either ``--gpus 0 1`` or ``--gpus 0,1`` notation."""
    if not values:
        return []
    return [gpu.strip() for value in values for gpu in value.split(",") if gpu.strip()]


def resolve_gpu_tokens(gpus: list[str], visible_devices: str | None = None) -> dict[str, str]:
    """Map process-logical indices back to Slurm's inherited CUDA tokens.

    A notebook inside ``CUDA_VISIBLE_DEVICES=3,5`` sees those devices as logical
    ``0,1``. Child workers must receive physical tokens ``3`` and ``5`` rather
    than accidentally escaping or conflicting with the allocation.
    """
    inherited_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible_devices is None else visible_devices
    inherited = [token.strip() for token in inherited_raw.split(",") if token.strip()]
    mapping: dict[str, str] = {}
    for gpu in gpus:
        if inherited and gpu.isdigit() and int(gpu) < len(inherited):
            mapping[gpu] = inherited[int(gpu)]
        else:
            mapping[gpu] = gpu
    return mapping
