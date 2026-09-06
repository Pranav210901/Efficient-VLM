from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.alignment_v3.fingerprint import hash_dataset_split, sha256_file
from src.phase15.io_utils import atomic_csv, atomic_json


def _score(seed: int, image_path: str) -> str:
    return hashlib.sha256(f"{seed}:{image_path}".encode("utf-8")).hexdigest()


def read_caption_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "caption"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain image_path and caption columns")
        rows = [{"image_path": str(row["image_path"]), "caption": str(row["caption"])} for row in reader]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def unique_image_ids(rows: Iterable[dict[str, str]]) -> list[str]:
    return sorted({row["image_path"] for row in rows})


def create_development_split(
    source_csv: str | Path,
    train_csv: str | Path,
    dev_csv: str | Path,
    manifest_path: str | Path,
    *,
    dev_images: int = 5000,
    seed: int = 3103,
    final_csv: str | Path | None = None,
) -> dict[str, object]:
    rows = read_caption_rows(source_csv)
    image_ids = unique_image_ids(rows)
    if dev_images <= 0 or dev_images >= len(image_ids):
        raise ValueError("dev_images must be positive and smaller than the training image count")
    ranked = sorted(image_ids, key=lambda value: (_score(seed, value), value))
    dev_ids = set(ranked[:dev_images])
    train_rows = [row for row in rows if row["image_path"] not in dev_ids]
    dev_rows = [row for row in rows if row["image_path"] in dev_ids]
    train_ids = unique_image_ids(train_rows)
    selected_dev_ids = unique_image_ids(dev_rows)
    if set(train_ids) & set(selected_dev_ids):
        raise AssertionError("train and development image IDs overlap")
    final_ids: list[str] = []
    if final_csv is not None:
        final_ids = unique_image_ids(read_caption_rows(final_csv))
        overlap = (set(train_ids) | set(selected_dev_ids)) & set(final_ids)
        if overlap:
            raise ValueError(f"COCO train/development overlaps final validation by {len(overlap)} images")
    atomic_csv(pd.DataFrame(train_rows), train_csv)
    atomic_csv(pd.DataFrame(dev_rows), dev_csv)
    manifest = {
        "status": "COMPLETE",
        "seed": seed,
        "source_csv": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "train_images": len(train_ids),
        "dev_images": len(selected_dev_ids),
        "train_captions": len(train_rows),
        "dev_captions": len(dev_rows),
        "train_split_hash": hash_dataset_split(train_ids),
        "dev_split_hash": hash_dataset_split(selected_dev_ids),
        "final_images": len(final_ids),
        "final_split_hash": hash_dataset_split(final_ids) if final_ids else None,
        "overlap_counts": {"train_dev": 0, "train_final": 0, "dev_final": 0},
    }
    atomic_json(manifest, manifest_path)
    return manifest


def validate_caption_alignment(rows: Iterable[dict[str, str]]) -> None:
    captions: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        captions[(row["image_path"], row["caption"])] += 1
    duplicates = sum(count - 1 for count in captions.values() if count > 1)
    if duplicates:
        raise ValueError(f"caption table contains {duplicates} duplicate image-caption rows")

