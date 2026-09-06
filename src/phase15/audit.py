"""Non-destructive inventory of completed Phase 1 and Phase 1.5 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import atomic_csv, atomic_json


ARTIFACTS = (
    "results/alignment_matrix_best_clean.csv",
    "results/seed_sweep_results.csv",
    "results/seed_sweep_summary.csv",
    "results/seed_sweep_paired_deltas.csv",
    "results/phase1_multitask",
    "results/phase15/predictions",
    "results/phase15/complementarity",
    "results/phase15/oracle",
    "results/phase15/cross_task",
    "results/phase15/efficiency",
    "results/phase15/hard_negatives",
    "results/phase15/expert_selection",
)


REQUIRED_COLUMNS = {
    "alignment_matrix_best_clean.csv": {"vision_encoder", "text_encoder", "variant"},
    "seed_sweep_results.csv": {"comparison", "vision_encoder", "text_encoder", "variant", "seed"},
    "seed_sweep_summary.csv": {"comparison", "variant", "n", "mean", "std"},
    "seed_sweep_paired_deltas.csv": {"comparison", "variant", "n", "mean_delta", "wins", "ties", "losses"},
    "evaluation_manifest.csv": {"config_id", "task", "status", "cache", "predictions"},
    "retrieval_pairwise.csv": {"expert_a", "expert_b"},
    "classification_pairwise.csv": {"dataset", "configuration_a", "configuration_b"},
    "retrieval_oracle.csv": {"expert_a", "expert_b", "oracle_R@1"},
    "classification_oracle.csv": {"dataset", "subset_id", "oracle_top1_accuracy"},
    "full_pairs.csv": {"config_id", "latency_ms", "peak_allocated_bytes"},
    "pair_shortlist.csv": {"config_id"},
}


def _safe_csv(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    try:
        return pd.read_csv(path), []
    except Exception as exc:
        return None, [f"{type(exc).__name__}: {exc}"]


def _directory_summary(path: Path) -> tuple[int, int, int, str, list[str]]:
    files = [value for value in path.rglob("*") if value.is_file()]
    issues: list[str] = []
    rows = 0
    configs: set[str] = set()
    tasks: set[str] = set()
    schema = "not_applicable"
    preferred = {
        "phase1_multitask": "evaluation_manifest.csv",
        "complementarity": "classification_pairwise.csv",
        "oracle": "classification_oracle.csv",
        "cross_task": "configuration_rankings.csv",
        "efficiency": "full_pairs.csv",
        "expert_selection": "pair_shortlist.csv",
    }.get(path.name)
    table = path / preferred if preferred else None
    if table is not None and table.exists():
        frame, read_issues = _safe_csv(table)
        issues.extend(read_issues)
        if frame is not None:
            rows = len(frame)
            for column in ("config_id", "configuration", "configuration_a", "expert_a"):
                if column in frame:
                    configs.update(frame[column].dropna().astype(str))
            if "configuration_b" in frame:
                configs.update(frame["configuration_b"].dropna().astype(str))
            if "task" in frame:
                tasks.update(frame["task"].dropna().astype(str))
            if "dataset" in frame:
                tasks.update(frame["dataset"].dropna().astype(str))
            required = REQUIRED_COLUMNS.get(table.name, set())
            missing = required - set(frame.columns)
            schema = "valid" if not missing else "invalid"
            if missing:
                issues.append(f"missing columns in {table.name}: {sorted(missing)}")
    elif table is not None:
        schema = "missing"
        issues.append(f"required derived table is missing: {table.name}")
    elif path.name == "predictions":
        rows = len(files)
        schema = "historical_mixed" if files else "missing"
    elif path.name == "hard_negatives":
        summary = path / "hard_negative_summary.json"
        if summary.exists():
            payload = json.loads(summary.read_text())
            rows = int(payload.get("retrieval_rows", 0)) + int(payload.get("classification_rows", 0))
            configs.update(payload.get("selected_config_ids", []))
            schema = "valid"
    if not files:
        issues.append("directory contains no files")
    return rows, len(configs), len(tasks), schema, issues


def run_phase15_audit(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    coverage_path = root / "results/phase15/prediction_validation/coverage_report.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    for artifact in ARTIFACTS:
        path = root / artifact
        exists = path.exists()
        issues: list[str] = []
        row_count = configuration_count = task_count = 0
        schema_status = "missing"
        fingerprint_status = "not_applicable"
        if exists and path.is_file():
            frame, issues = _safe_csv(path)
            if frame is not None:
                row_count = len(frame)
                for column in ("config_id", "run_name"):
                    if column in frame:
                        configuration_count = int(frame[column].nunique())
                        break
                if {"vision_encoder", "text_encoder", "variant"}.issubset(frame.columns):
                    configuration_count = int(frame[["vision_encoder", "text_encoder", "variant"]].drop_duplicates().shape[0])
                task_count = int(frame["task"].nunique()) if "task" in frame else 0
                required = REQUIRED_COLUMNS.get(path.name, set())
                missing = required - set(frame.columns)
                schema_status = "valid" if not missing else "invalid"
                if missing:
                    issues.append(f"missing columns: {sorted(missing)}")
        elif exists:
            row_count, configuration_count, task_count, schema_status, issues = _directory_summary(path)
        if artifact == "results/phase1_multitask" and not coverage.empty:
            fingerprint_status = "matched" if coverage["checkpoint_fingerprint_status"].isin(["matched", "matched_legacy_stat"]).all() else "mismatch_or_missing"
        valid = bool(exists and schema_status not in {"invalid", "missing"} and not issues)
        records.append(
            {
                "artifact": artifact,
                "exists": exists,
                "valid": valid,
                "row_count": row_count,
                "configuration_count": configuration_count,
                "task_count": task_count,
                "checkpoint_fingerprint_status": fingerprint_status,
                "schema_status": schema_status,
                "issues": "; ".join(issues),
            }
        )
    frame = pd.DataFrame(records)
    output = root / "results/phase15/audit"
    atomic_csv(frame, output / "phase15_artifact_audit.csv")
    atomic_json({"artifacts": json.loads(frame.to_json(orient="records")), "valid_count": int(frame["valid"].sum()), "artifact_count": len(frame)}, output / "phase15_artifact_audit.json")
    return frame
