from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml

from src.phase15.io_utils import atomic_json, atomic_text

from .config import fingerprint
from .schemas import RunStatus
from .schemas import REQUIRED_RUN_FILES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_payload(root: Path) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "git_commit": git_commit(root),
    }


def write_status(run_dir: str | Path, status: RunStatus | str, **details: Any) -> Path:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    value = status.value if isinstance(status, RunStatus) else str(status)
    return atomic_json({"status": value, "updated_at": utc_now(), **details}, target / "status.json")


def initialise_run(run_dir: str | Path, stage: str, config: dict[str, Any], root: str | Path) -> Path:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_config = {key: value for key, value in config.items() if key != "_config_path"}
    atomic_text(yaml.safe_dump(safe_config, sort_keys=False), destination / "run_config.yaml")
    atomic_json(environment_payload(Path(root).resolve()), destination / "environment.json")
    write_status(destination, RunStatus.RUNNING, stage=stage, config_fingerprint=fingerprint(safe_config), started_at=utc_now())
    return destination


@contextmanager
def managed_stage(run_dir: str | Path, stage: str, config: dict[str, Any], root: str | Path) -> Iterator[Path]:
    destination = initialise_run(run_dir, stage, config, root)
    try:
        yield destination
    except KeyboardInterrupt as exc:
        atomic_text(traceback.format_exc(), destination / "traceback.txt")
        write_status(destination, RunStatus.INTERRUPTED, stage=stage, error=str(exc))
        raise
    except Exception as exc:
        rendered = traceback.format_exc()
        atomic_text(rendered, destination / "traceback.txt")
        write_status(destination, RunStatus.FAILED, stage=stage, error=str(exc))
        print(rendered, file=sys.stderr)
        raise


def complete_stage(run_dir: str | Path, stage: str, summary: dict[str, Any]) -> None:
    destination = Path(run_dir)
    atomic_json(summary, destination / "summary.json")
    write_status(destination, RunStatus.COMPLETED, stage=stage, finished_at=utc_now())


def valid_completed_run(run_dir: str | Path, expected_fingerprint: str | None = None) -> bool:
    destination = Path(run_dir)
    try:
        status = json.loads((destination / "status.json").read_text())
        summary = json.loads((destination / "summary.json").read_text())
    except Exception:
        return False
    if status.get("status") != RunStatus.COMPLETED.value:
        return False
    if expected_fingerprint and summary.get("config_fingerprint") != expected_fingerprint:
        return False
    if not all((destination / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    checkpoint = summary.get("best_checkpoint")
    if checkpoint and not Path(checkpoint).exists():
        return False
    return True
