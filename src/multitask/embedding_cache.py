from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def save_cache(path: str | Path, payload: dict[str, Any], metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save({"metadata": metadata, "payload": payload}, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_cache(path: str | Path, expected_metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, ValueError):
        return None
    if not isinstance(value, dict) or "metadata" not in value or "payload" not in value:
        return None
    if expected_metadata is not None and value["metadata"] != expected_metadata:
        return None
    return value["payload"]
