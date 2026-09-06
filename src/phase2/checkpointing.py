from __future__ import annotations

import os
import random
import signal
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


class PreemptionState:
    requested = False
    signal_number: int | None = None

    @classmethod
    def handler(cls, signum, _frame) -> None:
        cls.requested = True
        cls.signal_number = int(signum)


def install_preemption_handlers() -> None:
    for value in (getattr(signal, "SIGUSR1", None), signal.SIGTERM):
        if value is not None:
            signal.signal(value, PreemptionState.handler)


def checkpoint_payload(model, optimizer=None, scheduler=None, scaler=None, **state: Any) -> dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        **state,
    }
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return payload


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _cpu_byte_rng_state(state: Any) -> torch.Tensor:
    """Return the CPU ByteTensor required by PyTorch RNG restoration APIs.

    Loading a checkpoint with ``map_location="cuda"`` also moves its saved RNG
    tensors to CUDA, while ``torch.set_rng_state`` and
    ``torch.cuda.set_rng_state_all`` require CPU byte tensors.  ``as_tensor``
    also keeps older checkpoints containing a list/array representation usable.
    """
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if payload.get("python_random_state") is not None: random.setstate(payload["python_random_state"])
    if payload.get("numpy_random_state") is not None: np.random.set_state(payload["numpy_random_state"])
    if payload.get("torch_random_state") is not None:
        torch.set_rng_state(_cpu_byte_rng_state(payload["torch_random_state"]))
    if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all([_cpu_byte_rng_state(state) for state in payload["cuda_random_state"]])
    return payload
