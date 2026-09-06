#!/usr/bin/env python3
"""Reversibly group completed artifact trees while preserving legacy paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PATH = ROOT / "experiments" / "layout.yaml"
RECEIPT_PATH = ROOT / "artifacts" / "layout_receipt.json"
KINDS = ("results", "checkpoints", "logs")
INTEGRITY_NAMES = {
    "config.yaml",
    "fingerprint.json",
    "report.json",
    "metrics.json",
    "run_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def layout() -> dict[str, Any]:
    return yaml.safe_load(LAYOUT_PATH.read_text(encoding="utf-8"))


def rows() -> list[dict[str, Path | str]]:
    spec = layout()
    canonical_root = ROOT / str(spec["canonical_root"])
    values: list[dict[str, Path | str]] = []
    for legacy_name, family in spec["families"].items():
        for kind in KINDS:
            source = ROOT / kind / legacy_name
            if source.exists() or source.is_symlink():
                values.append(
                    {
                        "legacy_name": str(legacy_name),
                        "family": str(family),
                        "kind": kind,
                        "source": source,
                        "target": canonical_root / str(family) / kind,
                    }
                )
    return values


def tree_stats(path: Path) -> dict[str, int]:
    files = [value for value in path.rglob("*") if value.is_file()]
    return {
        "files": len(files),
        "bytes": sum(value.stat().st_size for value in files),
        "device": path.stat().st_dev,
        "inode": path.stat().st_ino,
    }


def integrity_hashes(path: Path, legacy_prefix: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in sorted(path.rglob("*")):
        if value.is_file() and value.name in INTEGRITY_NAMES:
            relative = value.relative_to(path)
            hashes[str(legacy_prefix / relative)] = sha256(value)
    return hashes


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for value in path.rglob("*"):
        try:
            newest = max(newest, value.stat().st_mtime)
        except FileNotFoundError:
            continue
    return newest


def check(*, active_guard_minutes: int) -> dict[str, Any]:
    now = time.time()
    errors: list[str] = []
    planned: list[dict[str, Any]] = []
    already_grouped: list[str] = []

    for row in rows():
        source = row["source"]
        target = row["target"]
        assert isinstance(source, Path) and isinstance(target, Path)

        if source.is_symlink():
            if not target.is_dir() or source.resolve() != target.resolve():
                errors.append(f"invalid compatibility symlink: {source}")
            else:
                already_grouped.append(str(source.relative_to(ROOT)))
            continue
        if not source.is_dir():
            errors.append(f"source is not a directory: {source}")
            continue
        if target.exists() or target.is_symlink():
            errors.append(f"target already exists: {target}")
            continue
        age_minutes = (now - newest_mtime(source)) / 60.0
        if age_minutes < active_guard_minutes:
            errors.append(
                f"recent-write guard: {source.relative_to(ROOT)} changed "
                f"{age_minutes:.1f} minutes ago"
            )
            continue
        stats = tree_stats(source)
        planned.append(
            {
                "source": str(source.relative_to(ROOT)),
                "target": str(target.relative_to(ROOT)),
                "files": stats["files"],
                "bytes": stats["bytes"],
                "device": stats["device"],
                "inode": stats["inode"],
            }
        )

    return {
        "status": "READY" if not errors else "BLOCKED",
        "active_guard_minutes": active_guard_minutes,
        "planned": planned,
        "already_grouped": already_grouped,
        "errors": errors,
    }


def apply(*, active_guard_minutes: int) -> dict[str, Any]:
    audit = check(active_guard_minutes=active_guard_minutes)
    if audit["status"] != "READY":
        raise RuntimeError("layout check failed:\n- " + "\n- ".join(audit["errors"]))

    moved: list[dict[str, Any]] = []
    integrity_before: dict[str, str] = {}
    for row in rows():
        source = row["source"]
        target = row["target"]
        assert isinstance(source, Path) and isinstance(target, Path)
        if source.is_symlink():
            continue

        stats = tree_stats(source)
        integrity_before.update(integrity_hashes(source, source.relative_to(ROOT)))
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        relative_target = os.path.relpath(target, start=source.parent)
        source.symlink_to(relative_target, target_is_directory=True)

        if source.resolve() != target.resolve():
            raise RuntimeError(f"symlink verification failed for {source}")
        after = tree_stats(target)
        if (after["device"], after["inode"], after["files"], after["bytes"]) != (
            stats["device"],
            stats["inode"],
            stats["files"],
            stats["bytes"],
        ):
            raise RuntimeError(f"inode or tree-stat verification failed for {source}")
        moved.append(
            {
                "legacy_path": str(source.relative_to(ROOT)),
                "canonical_path": str(target.relative_to(ROOT)),
                **stats,
            }
        )

    integrity_after: dict[str, str] = {}
    for row in rows():
        source = row["source"]
        assert isinstance(source, Path)
        integrity_after.update(integrity_hashes(source, source.relative_to(ROOT)))
    if integrity_before != integrity_after:
        raise RuntimeError("integrity-file hashes changed during migration")

    receipt = {
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layout_sha256": sha256(LAYOUT_PATH),
        "legacy_path_policy": "relative_symlink",
        "protected_subtrees_untouched": layout()["protected_subtrees"],
        "moved": moved,
        "integrity_hashes": integrity_after,
    }
    atomic_json(receipt, RECEIPT_PATH)
    return receipt


def rollback() -> dict[str, Any]:
    if not RECEIPT_PATH.is_file():
        raise RuntimeError(f"missing migration receipt: {RECEIPT_PATH}")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    restored: list[str] = []
    for item in reversed(receipt["moved"]):
        source = ROOT / item["legacy_path"]
        target = ROOT / item["canonical_path"]
        if not source.is_symlink() or source.resolve() != target.resolve():
            raise RuntimeError(f"refusing rollback: unexpected legacy path {source}")
        source.unlink()
        target.rename(source)
        restored.append(item["legacy_path"])
    rollback_receipt = {
        "status": "ROLLED_BACK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "restored": restored,
    }
    atomic_json(rollback_receipt, RECEIPT_PATH.with_name("layout_rollback_receipt.json"))
    RECEIPT_PATH.unlink()
    return rollback_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", action="store_true")
    parser.add_argument("--active-guard-minutes", type=int, default=30)
    args = parser.parse_args()

    if args.rollback:
        payload = rollback()
    elif args.apply:
        payload = apply(active_guard_minutes=args.active_guard_minutes)
    else:
        payload = check(active_guard_minutes=args.active_guard_minutes)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

