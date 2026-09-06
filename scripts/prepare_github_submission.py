#!/usr/bin/env python3
"""Move non-submission material into a dated, recoverable quarantine.

The default mode is a dry run. Pass ``--apply`` to perform same-filesystem
moves and write a machine-readable manifest alongside the quarantined files.
Nothing is deleted and symlink targets are never followed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUARANTINE_NAME = "github_submission_cleanup_20260810"

FIXED_TARGETS = {
    ".cache": "downloaded model and package cache",
    ".venv": "re-creatable Python environment",
    ".venv-aisurrey": "re-creatable Python environment",
    ".pytest_cache": "generated test cache",
    "CHAT_TRANSCRIPT.md": "development transcript, not dissertation evidence",
    "graphify-out": "generated repository-analysis cache",
    "checkpoints": "compatibility symlinks to heavyweight checkpoints",
    "logs": "compatibility symlinks to raw training logs",
}

GENERATED_DIR_NAMES = {
    "__pycache__": "generated Python bytecode",
    ".ipynb_checkpoints": "generated notebook checkpoints",
    "checkpoints": "trained model checkpoints",
    "logs": "raw experiment logs",
    "teacher_cache": "generated teacher-feature cache",
    "feature_cache": "generated feature cache",
    "embeddings": "generated embedding tensors",
    "contact_sheets": "generated diagnostic contact sheets",
}

BINARY_SUFFIXES = {
    ".ckpt": "trained model binary",
    ".npy": "generated numerical array",
    ".npz": "generated numerical archive",
    ".pt": "trained model or tensor binary",
    ".pth": "trained model binary",
    ".safetensors": "trained model binary",
}

def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def discover_targets() -> dict[Path, str]:
    targets: dict[Path, str] = {}
    for rel, reason in FIXED_TARGETS.items():
        path = PROJECT_ROOT / rel
        if lexists(path):
            targets[path] = reason

    data_root = PROJECT_ROOT / "data"
    if data_root.is_dir():
        for child in data_root.iterdir():
            if child.name == "README.md":
                continue
            if child.name != "flickr30k":
                targets[child] = "local/raw dataset payload"
                continue
            for flickr_child in child.iterdir():
                if flickr_child.name not in {"validation.csv", "test.csv"}:
                    targets[flickr_child] = "local/raw Flickr30k payload"

    for current, dirnames, filenames in os.walk(PROJECT_ROOT, followlinks=False):
        current_path = Path(current)
        retained_dirs: list[str] = []
        for dirname in dirnames:
            candidate = current_path / dirname
            if current_path == PROJECT_ROOT and dirname in {"quarantine", ".git"}:
                continue
            if candidate in targets:
                continue
            if dirname in GENERATED_DIR_NAMES:
                targets[candidate] = GENERATED_DIR_NAMES[dirname]
            else:
                retained_dirs.append(dirname)
        dirnames[:] = retained_dirs
        for filename in filenames:
            candidate = current_path / filename
            reason = BINARY_SUFFIXES.get(candidate.suffix.lower())
            if reason:
                targets[candidate] = reason

    # If a parent is moving, listing its descendants separately only obscures
    # the manifest and can produce invalid move orderings.
    collapsed: dict[Path, str] = {}
    for path in sorted(targets, key=lambda item: len(item.parts)):
        if not any(parent in collapsed for parent in path.parents):
            collapsed[path] = targets[path]
    return collapsed


def describe(path: Path) -> tuple[str, int | None]:
    if path.is_symlink():
        return "symlink", None
    if path.is_dir():
        return "directory", None
    try:
        return "file", path.stat().st_size
    except OSError:
        return "file", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the moves (the default is a dry run)",
    )
    parser.add_argument(
        "--quarantine-name",
        default=DEFAULT_QUARANTINE_NAME,
        help="dated quarantine directory name under ./quarantine",
    )
    args = parser.parse_args()

    targets = discover_targets()
    quarantine_root = PROJECT_ROOT / "quarantine" / args.quarantine_name
    records: list[dict[str, object]] = []
    reasons = Counter(targets.values())

    print(f"project: {PROJECT_ROOT}")
    print(f"quarantine: {quarantine_root}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"targets: {len(targets)}")
    for reason, count in sorted(reasons.items()):
        print(f"  {count:4d}  {reason}")

    if not args.apply:
        return 0

    destination_root = quarantine_root / "original_paths"
    if quarantine_root.exists():
        raise SystemExit(f"refusing to reuse existing quarantine: {quarantine_root}")
    destination_root.mkdir(parents=True)

    moved_at = datetime.now(timezone.utc).isoformat()
    for source, reason in sorted(targets.items(), key=lambda item: str(item[0])):
        relative = source.relative_to(PROJECT_ROOT)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if lexists(destination):
            raise SystemExit(f"destination already exists: {destination}")
        kind, size_bytes = describe(source)
        os.rename(source, destination)
        records.append(
            {
                "original_path": str(relative),
                "quarantine_path": str(destination.relative_to(PROJECT_ROOT)),
                "kind": kind,
                "size_bytes": size_bytes,
                "reason": reason,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": moved_at,
        "project_root": str(PROJECT_ROOT),
        "policy": "Recoverable GitHub-submission cleanup; no files were deleted.",
        "restore": (
            "Move each quarantine_path back to original_path. Refuse to "
            "overwrite any path created after cleanup."
        ),
        "entries": records,
    }
    manifest_path = quarantine_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"moved: {len(records)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
