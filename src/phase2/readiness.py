from __future__ import annotations

import json
from pathlib import Path

from src.phase15.io_utils import atomic_json
from .config import parse_wall_time
from .prerequisites import validate_phase2_prerequisites


def write_implementation_readiness(project_root: str | Path, test_result: dict | None = None) -> dict:
    root = Path(project_root).resolve(); output = root / "results/phase2/readiness"; output.mkdir(parents=True, exist_ok=True)
    prerequisites = validate_phase2_prerequisites(root, write=False)
    scripts = sorted((root / "slurm/phase2").glob("*.sbatch")); violations = []
    import re
    for path in scripts:
        source = path.read_text()
        if "#SBATCH --partition=teaching" not in source: violations.append(f"{path.name}: partition")
        gpu = re.search(r"#SBATCH --gres=gpu:(\d+)", source)
        if gpu and int(gpu.group(1)) > 8: violations.append(f"{path.name}: GPUs")
        wall = re.search(r"#SBATCH --time=(\S+)", source)
        if not wall or parse_wall_time(wall.group(1)) > parse_wall_time("2-23:59:00"): violations.append(f"{path.name}: wall time")
    resource = {"status": "PASS" if not violations else "FAIL", "partition": "teaching", "maximum_gpus": 8, "maximum_wall_time": "2-23:59:00", "scripts_checked": len(scripts), "violations": violations}
    implementation = {
        "status": "IMPLEMENTATION_READY" if prerequisites["status"] == "VALID" and not violations else "NOT_READY",
        "prerequisites": prerequisites["status"], "phase15_readiness": prerequisites["phase15_readiness"],
        "selection_fingerprint": prerequisites["selection_sha256"], "primary_paths": prerequisites["primary_paths"],
        "notebook_smoke_only": True, "actual_execution": "sbatch_only", "phase3_implemented": False,
        "single_gpu_smoke": "READY_TO_SUBMIT", "two_gpu_ddp_smoke": "READY_TO_SUBMIT",
        "full_jobs_submitted_automatically": False,
    }
    tests = test_result or {"status": "NOT_RUN", "note": "Populate after pytest validation."}
    atomic_json(implementation, output / "implementation_report.json")
    atomic_json(resource, output / "resource_validation_report.json")
    atomic_json(tests, output / "test_report.json")
    return implementation
