#!/usr/bin/env python3
"""Validate the canonical experiment registry without changing any artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.yaml"
ALLOWED_STATUSES = {
    "COMPLETE",
    "STOPPED_BY_GATE",
    "INCOMPLETE_ARCHIVED",
    "SUPERSEDED",
    "READY_NOT_RUN",
    "IN_PROGRESS",
}


def main() -> int:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    experiments = data.get("experiments", [])
    errors: list[str] = []

    ids = [str(item.get("id", "")) for item in experiments]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate experiment IDs: {duplicate_ids}")

    covered_result_roots: set[str] = set()
    for item in experiments:
        experiment_id = item.get("id", "<missing-id>")
        status = item.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{experiment_id}: invalid status {status!r}")
        if not str(item.get("headline", "")).strip():
            errors.append(f"{experiment_id}: empty headline")

        for relative in item.get("physical_paths", []):
            path = ROOT / relative
            if not path.exists():
                errors.append(f"{experiment_id}: missing physical path {relative}")
            parts = Path(relative).parts
            if len(parts) >= 2 and parts[0] == "results":
                covered_result_roots.add(parts[1])

    actual_result_roots = {
        path.name for path in (ROOT / "results").iterdir() if path.is_dir()
    }
    unmapped = sorted(actual_result_roots - covered_result_roots)
    if unmapped:
        errors.append(f"unmapped top-level result directories: {unmapped}")

    if errors:
        print("Experiment registry: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    counts = Counter(item["status"] for item in experiments)
    print(
        "Experiment registry: PASS "
        f"({len(experiments)} entries; {len(actual_result_roots)} result roots)"
    )
    for status in sorted(counts):
        print(f"- {status}: {counts[status]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
