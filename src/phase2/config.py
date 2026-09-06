from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import deep_update


MAX_GPUS = 8
MAX_WALL_SECONDS = 2 * 86400 + 23 * 3600 + 59 * 60
ALLOWED_PARTITION = "teaching"


def project_root(value: str | Path | None = None) -> Path:
    return Path(value).resolve() if value is not None else Path(__file__).resolve().parents[2]


def load_phase2_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    parent = payload.pop("extends", None)
    if parent:
        inherited = Path(parent)
        if not inherited.is_absolute():
            inherited = source.parent / inherited
        payload = deep_update(load_phase2_config(inherited), payload)
    payload["_config_path"] = str(source)
    return payload


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(payload: Any, length: int = 20) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:length]


def validate_resource_request(resources: dict[str, Any]) -> dict[str, Any]:
    partition = str(resources.get("partition", ALLOWED_PARTITION))
    gpus = int(resources.get("gpus", 0))
    wall_seconds = parse_wall_time(str(resources.get("wall_time", "00:30:00")))
    if partition != ALLOWED_PARTITION:
        raise ValueError(f"Phase 2 partition must be {ALLOWED_PARTITION!r}")
    if gpus < 0 or gpus > MAX_GPUS:
        raise ValueError(f"GPU request must be between 0 and {MAX_GPUS}")
    if wall_seconds > MAX_WALL_SECONDS:
        raise ValueError("Wall time exceeds 2-23:59:00")
    return {"partition": partition, "gpus": gpus, "wall_seconds": wall_seconds}


def parse_wall_time(value: str) -> int:
    days = 0
    clock = value
    if "-" in value:
        day, clock = value.split("-", 1)
        days = int(day)
    parts = [int(item) for item in clock.split(":")]
    if len(parts) != 3 or any(item < 0 for item in parts) or parts[1] >= 60 or parts[2] >= 60:
        raise ValueError(f"Invalid Slurm wall time: {value}")
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def safe_array_concurrency(gpus_per_task: int, max_total_gpus: int = MAX_GPUS) -> int:
    if not 1 <= gpus_per_task <= MAX_GPUS:
        raise ValueError("gpus_per_task must be in [1,8]")
    if not 1 <= max_total_gpus <= MAX_GPUS:
        raise ValueError("max_total_gpus must be in [1,8]")
    value = max_total_gpus // gpus_per_task
    if value < 1:
        raise ValueError("GPU budget cannot accommodate one task")
    return value
