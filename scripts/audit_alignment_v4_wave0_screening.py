#!/usr/bin/env python3
"""Read-only integrity audit of the 18 Wave 0 screening cells."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alignment_v3.fingerprint import hash_config


RESULTS = ROOT / "results/alignment_v4_wave0/wave0-screen"
CHECKPOINTS = ROOT / "checkpoints/alignment_v4_wave0/wave0-screen"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fingerprint_digest(value: Any) -> str | None:
    if isinstance(value, dict):
        return str(value.get("digest")) if value.get("digest") else None
    return None


def audit_cell(result_dir: Path) -> dict[str, Any]:
    run_id = result_dir.name
    checkpoint_dir = CHECKPOINTS / run_id
    paths = {
        "metrics": result_dir / "metrics.json",
        "result_fingerprint": result_dir / "fingerprint.json",
        "config": checkpoint_dir / "config.yaml",
        "checkpoint_fingerprint": checkpoint_dir / "fingerprint.json",
        "best": checkpoint_dir / "best.pt",
        "run_summary": checkpoint_dir / "run_summary.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {"run_id": run_id, "status": "MISSING", "missing": missing}

    metrics = _load_json(paths["metrics"])
    result_fingerprint = _load_json(paths["result_fingerprint"])
    checkpoint_fingerprint = _load_json(paths["checkpoint_fingerprint"])
    config = yaml.safe_load(paths["config"].read_text())
    summary = _load_json(paths["run_summary"])
    best = torch.load(paths["best"], map_location="cpu", weights_only=False)
    best_config = dict(best["config"])
    claimed_checkpoint = ROOT / str(metrics["checkpoint"])
    claimed = (
        torch.load(claimed_checkpoint, map_location="cpu", weights_only=False)
        if claimed_checkpoint.is_file()
        else None
    )

    directory_lr = float(config["training"]["lr"])
    checkpoint_lr = float(best_config["training"]["lr"])
    claimed_lr = (
        float(claimed["config"]["training"]["lr"]) if claimed is not None else None
    )
    metric_lr = metrics.get("resolved_lr")
    result_digest = str(result_fingerprint["digest"])
    checkpoint_digest = str(checkpoint_fingerprint["digest"])
    embedded_digest = _fingerprint_digest(best.get("fingerprint"))
    claimed_digest = (
        _fingerprint_digest(claimed.get("fingerprint")) if claimed is not None else None
    )
    expected_checkpoint = paths["best"].relative_to(ROOT)

    config_mtime = paths["config"].stat().st_mtime
    best_mtime = paths["best"].stat().st_mtime
    summary_mtime = paths["run_summary"].stat().st_mtime
    observed_window = summary_mtime - config_mtime
    wall_seconds = float(summary["wall_seconds"])
    mtime_consistent = (
        config_mtime <= best_mtime <= summary_mtime + 5.0
        and observed_window + 5.0 >= wall_seconds
    )

    discrepancies: list[str] = []
    warnings: list[str] = []
    if metric_lr is None:
        warnings.append("metrics_resolved_lr_missing")
    elif abs(float(metric_lr) - directory_lr) > 1e-15:
        discrepancies.append("metrics_resolved_lr_vs_directory_config")
    if abs(directory_lr - checkpoint_lr) > 1e-15:
        discrepancies.append("directory_config_vs_directory_checkpoint_lr")
    if claimed_lr is None:
        discrepancies.append("metrics_claimed_checkpoint_missing")
    elif abs(directory_lr - claimed_lr) > 1e-15:
        discrepancies.append("directory_config_vs_metrics_claimed_checkpoint_lr")
    if str(metrics["checkpoint"]) != str(expected_checkpoint):
        discrepancies.append("metrics_points_outside_cell_checkpoint_directory")
    if hash_config(config) != checkpoint_fingerprint["config_hash"]:
        discrepancies.append("directory_config_hash_vs_checkpoint_fingerprint")
    if checkpoint_digest != embedded_digest:
        discrepancies.append("directory_checkpoint_vs_embedded_fingerprint")
    if str(metrics["fingerprint_digest"]) != result_digest:
        discrepancies.append("metrics_vs_result_fingerprint")
    if result_digest != checkpoint_digest:
        discrepancies.append("result_vs_directory_checkpoint_fingerprint")
    if claimed_digest is not None and result_digest != claimed_digest:
        discrepancies.append("result_vs_claimed_checkpoint_fingerprint")
    if not mtime_consistent:
        discrepancies.append("checkpoint_mtime_vs_run_summary_wall_seconds")

    return {
        "run_id": run_id,
        "status": "DISAGREEMENT" if discrepancies else "CONSISTENT",
        "directory_resolved_lr": directory_lr,
        "metrics_resolved_lr": metric_lr,
        "claimed_checkpoint_resolved_lr": claimed_lr,
        "metrics_checkpoint": str(metrics["checkpoint"]),
        "expected_checkpoint": str(expected_checkpoint),
        "wall_seconds": wall_seconds,
        "mtime_window_seconds": observed_window,
        "mtime_consistent": mtime_consistent,
        "fingerprints": {
            "metrics": str(metrics["fingerprint_digest"]),
            "result": result_digest,
            "directory_checkpoint": checkpoint_digest,
            "directory_embedded": embedded_digest,
            "claimed_embedded": claimed_digest,
        },
        "discrepancies": discrepancies,
        "warnings": warnings,
    }


def main() -> None:
    rows = [audit_cell(path) for path in sorted(RESULTS.iterdir()) if path.is_dir()]
    disagreements = [row for row in rows if row["status"] != "CONSISTENT"]
    print(
        json.dumps(
            {
                "status": "DISAGREEMENTS_FOUND" if disagreements else "CONSISTENT",
                "cells_audited": len(rows),
                "disagreement_count": len(disagreements),
                "warning_count": sum(len(row.get("warnings", [])) for row in rows),
                "disagreements": disagreements,
                "cells": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
