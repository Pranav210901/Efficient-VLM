"""Availability report for optional, evaluation-only compositional tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import atomic_csv, atomic_json


DATASET_CANDIDATES = {
    "winoground": ("data/multitask/winoground", "data/winoground"),
    "sugarcrepe": ("data/multitask/sugarcrepe", "data/sugarcrepe"),
}


def run_compositional_status(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for task, candidates in DATASET_CANDIDATES.items():
        locations = [root / value for value in candidates]
        available = next((value for value in locations if value.exists()), None)
        prediction_files = sorted((root / "results/phase1_multitask/predictions").glob(f"*{task}*.jsonl"))
        if available is None:
            status, issue = "missing", "dataset is not installed; no automatic download attempted"
        elif not any(available.rglob("*")):
            status, issue = "invalid", "dataset directory is empty"
        elif prediction_files:
            status, issue = "ready", "sample-level evaluation outputs are available"
        else:
            status, issue = "evaluation_only", "dataset is local but evaluation outputs have not been generated"
        rows.append(
            {
                "task": task,
                "status": status,
                "dataset_path": str(available) if available else "",
                "prediction_files": len(prediction_files),
                "allowed_for_training": False,
                "usage": "evaluation_only",
                "issues": issue,
            }
        )
    frame = pd.DataFrame(rows)
    output = root / "results/phase15/compositional"
    atomic_csv(frame, output / "task_status.csv")
    atomic_json({"tasks": frame.to_dict("records"), "mandatory_for_phase15": False}, output / "task_status.json")
    return frame
