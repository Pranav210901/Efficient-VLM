from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.phase15.io_utils import atomic_json


def token_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def save_token_cache(path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"tensors": {key: value.detach().cpu() for key, value in tensors.items()}, "metadata": metadata}, temporary)
    temporary.replace(target)
    atomic_json(metadata, target.with_suffix(target.suffix + ".json"))
    return target


def load_token_cache(path: str | Path, expected: dict[str, Any] | None = None) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if expected is not None and payload.get("metadata") != expected:
        return None
    return payload
