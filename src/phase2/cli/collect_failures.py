from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.phase2.status import utc_now


def _sacct(job_ids: list[str]) -> pd.DataFrame:
    if not job_ids: return pd.DataFrame()
    try:
        output = subprocess.check_output(["sacct", "-n", "-P", "-j", ",".join(job_ids), "--format=JobIDRaw,State,ExitCode,Elapsed"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return pd.DataFrame()
    rows = [line.split("|")[:4] for line in output.splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=["job_id", "slurm_state", "exit_code", "runtime"])


def collect(root: Path, failing_stage: str | None = None, failing_job: str | None = None) -> dict:
    destination = root / "results/phase2/slurm"; destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": []}
    jobs = pd.DataFrame(manifest.get("stages", []))
    ids = [str(value) for value in jobs.get("job_id", []) if str(value).isdigit()]
    accounting = _sacct(ids)
    if not jobs.empty and not accounting.empty: jobs = jobs.merge(accounting, on="job_id", how="left")
    status_rows = []
    for row in manifest.get("stages", []):
        status_path = root / row.get("status_path", "") if row.get("status_path") else None
        state = "MISSING"
        observed = []; observed_indices = []
        if status_path and status_path.exists():
            candidates = [status_path] if status_path.is_file() else sorted(status_path.glob("*/status.json"))
            for candidate in candidates:
                try: observed.append(json.loads(candidate.read_text()).get("status", "INVALID")); observed_indices.append(candidate.parent.name)
                except Exception: observed.append("INVALID")
            if observed:
                state = "COMPLETED" if all(value == "COMPLETED" for value in observed) else next((value for value in observed if value != "COMPLETED"), "INVALID")
        expected = [root / value for value in row.get("expected_outputs", [])]
        missing_outputs = [str(value) for value in expected if not value.exists() or (value.is_dir() and not any(value.iterdir()))]
        expected_indices = {str(index) for index in range(int(row.get("array_size", 1)))} if int(row.get("array_size", 1)) > 1 else {"main"}
        missing_indices = sorted(expected_indices.difference(observed_indices))
        status_rows.append({**row, "status_file_state": state, "output_valid": not missing_outputs, "missing_outputs": ";".join(missing_outputs), "failed_array_indices": ";".join(missing_indices), "status_path": str(status_path) if status_path else ""})
    status = pd.DataFrame(status_rows)
    failed = status[status["status_file_state"].isin(["FAILED", "INTERRUPTED", "INVALID", "MISSING"]) | ~status["output_valid"]] if not status.empty else status
    atomic_csv(status, destination / "pipeline_status.csv"); atomic_csv(failed, destination / "failed_jobs.csv")
    summary = {"created_at": utc_now(), "failing_stage": failing_stage, "failing_job": failing_job, "completed_stages": int(status["status_file_state"].eq("COMPLETED").sum()) if not status.empty else 0, "failed_or_missing_stages": int(len(failed)), "suggested_resume_command": "bash scripts/phase2_resume.sh"}
    atomic_json(summary, destination / "pipeline_status.json")
    markdown = "# Phase 2 Pipeline Failure Report\n\n" + f"Generated: {summary['created_at']}\n\n" + (failed.to_markdown(index=False) if not failed.empty else "No failed stages were identified.") + "\n\nResume with: `bash scripts/phase2_resume.sh`\n"
    atomic_text(markdown, destination / "failure_report.md"); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=None); parser.add_argument("--stage"); parser.add_argument("--job-id")
    args = parser.parse_args(); root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[3]
    print(collect(root, args.stage, args.job_id))


if __name__ == "__main__": main()
