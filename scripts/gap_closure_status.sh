#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=".venv-aisurrey/bin/python"
"${python_bin}" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()

def state(path: str, complete_key: str = "status"):
    value = root / path
    if not value.is_file():
        return {"state": "MISSING", "path": path}
    try:
        payload = json.loads(value.read_text())
    except Exception as exc:
        return {"state": "INVALID", "path": path, "detail": str(exc)}
    return {"state": payload.get(complete_key, "PRESENT"), "path": path}

annotation_root = root / "data/multitask/sugarcrepe/annotations"
winoground = root / "data/multitask/winoground/data/test-00000-of-00001.parquet"
status = {
    "bootstrap": state("artifacts/07_final_evaluation/uncertainty/report.json"),
    "sugarcrepe": {
        "state": "READY" if len(list(annotation_root.glob("*.json"))) == 7 else "MISSING",
        "annotation_files": len(list(annotation_root.glob("*.json"))),
    },
    "winoground": {
        "state": "READY" if winoground.is_file() else "OPTIONAL_NOT_ACQUIRED",
        "path": str(winoground.relative_to(root)),
    },
    "compositional_freeze": state("artifacts/07_final_evaluation/compositional/manifests/data_manifest.json"),
    "compositional_results": state("artifacts/07_final_evaluation/compositional/report/report.json"),
    "queue_design": state("results/queue_mechanism/manifests/validation.json"),
    "queue_results": state("results/queue_mechanism/report/report.json"),
}
print(json.dumps(status, indent=2))
PY
