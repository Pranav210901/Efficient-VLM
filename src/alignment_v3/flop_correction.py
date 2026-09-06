from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.alignment_v3.runner import ROOT
from src.phase15.io_utils import atomic_csv, atomic_json


AUDIT_ROOT = ROOT / "results/efficiency_frontier/flop_correction"
SNAPSHOT = AUDIT_ROOT / "pre_correction_snapshot.json"


def _artifacts() -> list[Path]:
    values = list(
        (ROOT / "results/efficiency_frontier/per_run").glob(
            "**/profile*.json"
        )
    )
    values += [
        ROOT / "results/efficiency_frontier/controls/mobileclip_fusion_profile.json"
    ]
    values += list(
        (ROOT / "results/resolution_arm/profiling/per_run").glob(
            "**/profile_dynamic_padding.json"
        )
    )
    return sorted({path for path in values if path.is_file()})


def snapshot() -> dict[str, Any]:
    rows = []
    for path in _artifacts():
        payload = json.loads(path.read_text())
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "entry_id": payload.get("entry_id")
                or payload.get("experiment_id")
                or path.parent.name,
                "seed": payload.get("seed"),
                "resolution": payload.get("native_image_resolution")
                or payload.get("image_resolution"),
                "old_image_flops": payload.get("image_flops"),
                "old_caption_flops": payload.get("caption_flops"),
                "old_flop_source": payload.get("flop_source"),
            }
        )
    result = {
        "status": "SNAPSHOT_COMPLETE",
        "artifacts": len(rows),
        "rows": rows,
    }
    atomic_json(result, SNAPSHOT)
    return result


def report() -> dict[str, Any]:
    if not SNAPSHOT.is_file():
        raise FileNotFoundError("pre-correction FLOP snapshot is missing")
    old_rows = {
        row["path"]: row for row in json.loads(SNAPSHOT.read_text())["rows"]
    }
    rows = []
    for path_text, old in old_rows.items():
        path = ROOT / path_text
        if not path.is_file():
            raise FileNotFoundError(path)
        new = json.loads(path.read_text())
        if new.get("flop_measurement_version") != "operator_level_v2":
            raise RuntimeError(f"artifact was not corrected: {path}")
        old_image = (
            float(old["old_image_flops"])
            if old["old_image_flops"] is not None
            else None
        )
        new_image = float(new["image_flops"])
        old_text = (
            float(old["old_caption_flops"])
            if old["old_caption_flops"] is not None
            else None
        )
        new_text = float(new["caption_flops"])
        rows.append(
            {
                **old,
                "new_image_flops": new_image,
                "image_absolute_error_flops": (
                    old_image - new_image if old_image is not None else None
                ),
                "image_old_over_new_ratio": (
                    old_image / new_image if old_image is not None else None
                ),
                "new_caption_flops": new_text,
                "caption_absolute_error_flops": (
                    old_text - new_text if old_text is not None else None
                ),
                "caption_old_over_new_ratio": (
                    old_text / new_text if old_text is not None else None
                ),
                "new_flop_source": new["flop_source"],
                "old_value_wrong": (
                    old_image is None
                    or old_text is None
                    or old_image != new_image
                    or old_text != new_text
                ),
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, AUDIT_ROOT / "corrections.csv")
    result = {
        "status": "COMPLETE",
        "artifacts_checked": len(rows),
        "artifacts_with_wrong_stored_flops": int(frame["old_value_wrong"].sum()),
        "all_operator_level_v2": True,
        "corrections_csv": str(
            (AUDIT_ROOT / "corrections.csv").relative_to(ROOT)
        ),
    }
    atomic_json(result, AUDIT_ROOT / "report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("snapshot", "report"))
    args = parser.parse_args()
    result = snapshot() if args.command == "snapshot" else report()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
