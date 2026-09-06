from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.phase15.io_utils import atomic_csv, atomic_json, sha256_file
from src.phase2.ablations import build_phase2_manifests
from src.phase2.config import fingerprint, load_phase2_config, safe_array_concurrency, validate_resource_request
from src.phase2.prerequisites import validate_phase2_prerequisites
from src.phase2.status import git_commit, utc_now
import pandas as pd


STAGES = [
    ("00_validate_prerequisites", "prerequisites.yaml", None, 0),
    ("01_validate_token_interfaces", "token_interfaces.yaml", None, 1),
    ("02_build_training_data", "data_build.yaml", None, 0),
    ("03_smoke_cross_attention", "smoke.yaml", None, 1),
    ("03b_smoke_ddp", "smoke.yaml", None, 2),
    ("04_train_bridges_array", "bridge_training.yaml", 4, 1),
    ("05_evaluate_bridges_array", "bridge_evaluation.yaml", 4, 1),
    ("06_run_ablations_array", "ablations.yaml", 22, 1),
    ("07_train_dense_teacher", "dense_teacher.yaml", None, 4),
    ("08_evaluate_dense_teacher", "dense_teacher.yaml", None, 1),
    ("09_seed_sweep_array", "seed_sweep.yaml", 10, 1),
    ("10_build_phase2_report", "reporting.yaml", None, 0),
]

EXPECTED_OUTPUTS = {
    "00_validate_prerequisites": ["results/phase2/prerequisites/prerequisite_report.json", "results/phase2/prerequisites/validated_selection.yaml"],
    "01_validate_token_interfaces": ["results/phase2/token_interfaces/token_interface_manifest.csv"],
    "02_build_training_data": ["results/phase2/data/schema.json", "results/phase2/data/leakage_report.json"],
    "03_smoke_cross_attention": ["results/phase2/smoke/synthetic_smoke.json"],
    "03b_smoke_ddp": ["results/phase2/smoke/ddp_smoke.json"],
    "04_train_bridges_array": ["results/phase2/bridges/checkpoints", "results/phase2/bridges/runs"],
    "05_evaluate_bridges_array": ["results/phase2/reranking/runs"],
    "06_run_ablations_array": ["results/phase2/ablations/runs"],
    "07_train_dense_teacher": ["results/phase2/dense_teacher/checkpoints"],
    "08_evaluate_dense_teacher": ["results/phase2/dense_teacher/results_best.csv"],
    "09_seed_sweep_array": ["results/phase2/seed_sweep/runs"],
    "10_build_phase2_report": ["results/phase2/phase2_report.json"],
}


def _submit(command: list[str], dry_run: bool, fake: int) -> str:
    print(" ".join(command), flush=True)
    if dry_run: return f"DRY{fake:04d}"
    return subprocess.check_output(command, text=True).strip().split(";")[0]


def _stage_completed(root: Path, stage: tuple[str, str, int | None, int]) -> bool:
    name, _, array_size, _ = stage; status_root = root / "results/phase2/slurm_status" / name
    expected = array_size or 1
    statuses = sorted(status_root.glob("*/status.json")) if status_root.exists() else []
    if len(statuses) < expected: return False
    try:
        valid_status = all(json.loads(path.read_text()).get("status") == "COMPLETED" for path in statuses)
    except Exception:
        return False
    outputs = [root / value for value in EXPECTED_OUTPUTS[name]]
    return valid_status and all(path.exists() and (not path.is_dir() or any(path.iterdir())) for path in outputs)


def _incomplete_array_indices(root: Path, stage: str, array_size: int) -> list[int]:
    status_root = root / "results/phase2/slurm_status" / stage
    incomplete = []
    for index in range(array_size):
        status_path = status_root / str(index) / "status.json"
        try:
            completed = json.loads(status_path.read_text()).get("status") == "COMPLETED"
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            completed = False
        if not completed:
            incomplete.append(index)
    return incomplete


def submit_pipeline(root: Path, args) -> dict[str, Any]:
    prerequisite = validate_phase2_prerequisites(root, write=False)
    if prerequisite["status"] != "VALID": raise RuntimeError("Phase 1.5 prerequisites are not READY/valid")
    build_phase2_manifests(root)
    config_root = Path(args.config_root); config_root = config_root if config_root.is_absolute() else root / config_root
    if not 1 <= args.max_total_gpus <= 8: raise ValueError("--max-total-gpus must be in [1,8]")
    first = next((i for i, stage in enumerate(STAGES) if stage[0] == args.resume_from), 0) if args.resume_from else 0
    if args.stage:
        selected = [value for value in STAGES if value[0] == args.stage]
        if not selected: raise ValueError(f"Unknown stage {args.stage}")
    else: selected = STAGES[first:]
    forced = set(args.force_stage or [])
    unknown_forced = forced.difference(stage[0] for stage in STAGES)
    if unknown_forced: raise ValueError(f"Unknown forced stages: {sorted(unknown_forced)}")
    if args.skip_completed: selected = [stage for stage in selected if stage[0] in forced or not _stage_completed(root, stage)]
    stage_rows, previous = [], None; all_ids = []
    for position, (stage, config_name, array_size, default_gpus) in enumerate(selected, 1):
        config_path = config_root / config_name; config = load_phase2_config(config_path); resources = config.get("resources", {})
        request = validate_resource_request(resources); gpus = args.gpus if args.gpus is not None and default_gpus else int(default_gpus)
        if default_gpus and gpus < 1: raise ValueError(f"{stage} requires at least one GPU")
        if gpus > 8: raise ValueError("No Phase 2 job may request more than eight GPUs")
        (root / "logs/phase2" / stage).mkdir(parents=True, exist_ok=True)
        (root / "logs/phase2/99_report_failure").mkdir(parents=True, exist_ok=True)
        (root / "logs/phase2/90_collect_pipeline_status").mkdir(parents=True, exist_ok=True)
        command = ["sbatch", "--parsable"]
        if gpus: command.append(f"--gres=gpu:{gpus}")
        if previous: command.append(f"--dependency=afterok:{previous}")
        if array_size:
            throttle = safe_array_concurrency(max(1, gpus), args.max_total_gpus)
            array_indices = _incomplete_array_indices(root, stage, array_size) if args.skip_completed else list(range(array_size))
            # If statuses claim completion but shared outputs are missing, rerun
            # the whole array because the missing producer cannot be identified.
            if not array_indices:
                array_indices = list(range(array_size))
            array_spec = ",".join(map(str, array_indices))
            command.append(f"--array={array_spec}%{throttle}")
        else:
            array_indices = None
        command.extend([f"--export=ALL,PHASE2_CONFIG_ROOT={config_root},PHASE2_GPUS={gpus}", str(root / f"slurm/phase2/{stage}.sbatch")])
        job_id = _submit(command, args.dry_run, position); all_ids.append(job_id)
        failure = _submit(["sbatch", "--parsable", f"--dependency=afternotok:{job_id}", f"--export=ALL,FAILED_STAGE={stage},FAILED_JOB_ID={job_id}", str(root / "slurm/phase2/99_report_failure.sbatch")], args.dry_run, 100 + position)
        all_ids.append(failure)
        stage_rows.append({"stage": stage, "job_id": job_id, "failure_job_id": failure, "depends_on": previous, "config": str(config_path), "config_fingerprint": fingerprint(config), "partition": "teaching", "gpus": gpus, "cpus": resources.get("cpus"), "memory": resources.get("memory"), "wall_time": resources.get("wall_time"), "array_size": array_size or 1, "submitted_array_indices": array_indices, "expected_outputs": EXPECTED_OUTPUTS[stage], "status_path": f"results/phase2/slurm_status/{stage}", "log_path": f"logs/phase2/{stage}"})
        previous = job_id
    collector_dependency = ":".join(all_ids)
    collector = _submit(["sbatch", "--parsable", f"--dependency=afterany:{collector_dependency}", str(root / "slurm/phase2/90_collect_pipeline_status.sbatch")], args.dry_run, 999) if all_ids else None
    destination = root / "results/phase2/slurm"; destination.mkdir(parents=True, exist_ok=True)
    prefix = "dry_run_" if args.dry_run else ""
    manifest = {"submission_timestamp": utc_now(), "dry_run": args.dry_run, "git_commit": git_commit(root), "selection_fingerprint": prerequisite["selection_sha256"], "stages": stage_rows, "final_status_job_id": collector}
    atomic_json(manifest, destination / f"{prefix}submission_manifest.json")
    atomic_csv(pd.DataFrame(stage_rows), destination / f"{prefix}job_ids.csv")
    atomic_json({"maximum_total_gpus": args.max_total_gpus, "partition": "teaching", "stages": [{"stage": row["stage"], "gpus": row["gpus"], "array_size": row["array_size"]} for row in stage_rows]}, destination / f"{prefix}resource_plan.json")
    atomic_json({"edges": [{"from": row["depends_on"], "to": row["job_id"], "condition": "afterok"} for row in stage_rows if row["depends_on"]], "failure_edges": [{"from": row["job_id"], "to": row["failure_job_id"], "condition": "afternotok"} for row in stage_rows]}, destination / f"{prefix}dependency_graph.json")
    atomic_json({"command": "bash scripts/phase2_resume.sh", "skip_completed": True}, destination / f"{prefix}resume_plan.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=None); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--max-total-gpus", type=int, default=8); parser.add_argument("--gpus", type=int); parser.add_argument("--stage"); parser.add_argument("--resume-from"); parser.add_argument("--skip-completed", action="store_true"); parser.add_argument("--force-stage", action="append", default=[]); parser.add_argument("--config-root", default="configs/phase2")
    args = parser.parse_args(); root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    print(json.dumps(submit_pipeline(root, args), indent=2))


if __name__ == "__main__": main()
