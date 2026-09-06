#!/usr/bin/env python3
"""Record the post-observation reporting amendment without touching runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "freezeshift"
DESIGN = STUDY_ROOT / "results/manifests/design.json"
PRE_AMENDMENT = STUDY_ROOT / "results/manifests/design_pre_reporting_amendment.json"
RECEIPT = STUDY_ROOT / "results/manifests/reporting_amendment.json"
ORIGINAL_PACKAGE_SHA256 = "613084e24aa468925506a6fd38f3e527da3155a7ce6cda4cc1b1a1669b192988"


def package_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in STUDY_ROOT.rglob("*")
        if path.is_file()
        and not any(
            part in {"results", "checkpoints", "logs", "__pycache__"}
            for part in path.parts
        )
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    design = json.loads(DESIGN.read_text())
    recorded = str(design["package_sha256"])
    if recorded not in {ORIGINAL_PACKAGE_SHA256, package_hash()}:
        raise RuntimeError(
            "refusing amendment: design manifest does not contain the original "
            "or current package hash"
        )
    if not PRE_AMENDMENT.exists():
        atomic_json(design, PRE_AMENDMENT)
    current = package_hash()
    amendment = {
        "status": "COMPLETE",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "scope": "reporting_only",
        "reason": (
            "Dual-arm seed 43 tied exactly at Flickr-validation epochs 23 and 24; "
            "the original strict reporter stopped because no tie-break was registered."
        ),
        "tie_break": "earliest_epoch",
        "tie_break_status": "post_observation_reporting_amendment",
        "affected_observed_tie": {
            "arm": "dual",
            "seed": 43,
            "epochs": [23, 24],
            "mean_R@1": 0.6313609480857849,
            "selected_epoch": 23,
        },
        "baseline_correction": {
            "stale_inherited_flickr_validation_mean_R1": 0.44714,
            "matched_M_T1_flickr_validation_mean_R1": 0.5448717921972275,
            "matched_M_T1_repeated_pooled_q3_ms": 8.99804092478007,
        },
        "training_artifacts_changed": False,
        "evaluation_artifacts_changed": False,
        "flickr_test_remained_sealed": True,
        "original_package_sha256": ORIGINAL_PACKAGE_SHA256,
        "amended_package_sha256": current,
    }
    design["package_sha256"] = current
    design["post_observation_reporting_amendment"] = amendment
    atomic_json(amendment, RECEIPT)
    atomic_json(design, DESIGN)
    print(json.dumps(amendment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
