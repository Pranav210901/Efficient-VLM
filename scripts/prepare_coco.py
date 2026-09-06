from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from random import Random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO Captions annotations into project CSV files.")
    parser.add_argument("--coco_dir", default="data/coco", help="Directory containing train2017, val2017, and annotations.")
    parser.add_argument("--train_json", default=None, help="Override path to captions_train2017.json.")
    parser.add_argument("--val_json", default=None, help="Override path to captions_val2017.json.")
    parser.add_argument("--train_image_dir", default=None, help="Override path to COCO train image directory.")
    parser.add_argument("--val_image_dir", default=None, help="Override path to COCO validation image directory.")
    parser.add_argument("--output_train_csv", default="data/train.csv")
    parser.add_argument("--output_val_csv", default="data/val.csv")
    parser.add_argument("--all_captions", action="store_true", help="Write every COCO caption instead of one caption per image.")
    parser.add_argument("--all_val_captions", action="store_true", help="Write all validation captions while keeping one training caption per image.")
    parser.add_argument("--split", choices=["both", "train", "val"], default="both", help="Prepare both splits or only one split.")
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--max_val", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_rows(
    annotation_path: Path,
    image_dir: Path,
    *,
    all_captions: bool,
    max_rows: int | None,
    seed: int,
) -> list[dict[str, str]]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"COCO image directory not found: {image_dir}")

    payload: dict[str, Any] = json.loads(annotation_path.read_text())
    images = {int(image["id"]): str(image["file_name"]) for image in payload.get("images", [])}
    annotations = payload.get("annotations", [])
    if not images or not annotations:
        raise ValueError(f"{annotation_path} does not look like a COCO Captions annotation file")

    grouped: dict[int, list[str]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        caption = str(annotation["caption"]).strip()
        if image_id in images and caption:
            grouped.setdefault(image_id, []).append(caption)

    rng = Random(seed)
    rows: list[dict[str, str]] = []
    for image_id in sorted(grouped):
        file_name = images[image_id]
        image_path = image_dir / file_name
        if not image_path.exists():
            continue

        captions = grouped[image_id] if all_captions else [grouped[image_id][0]]
        rel_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        rows.extend({"image_path": rel_path, "caption": caption} for caption in captions)

    if max_rows is not None:
        rng.shuffle(rows)
        rows = rows[:max_rows]
        rows.sort(key=lambda row: row["image_path"])

    if not rows:
        raise ValueError(f"No usable image-caption rows were produced from {annotation_path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    coco_dir = resolve_path(args.coco_dir)
    train_json = resolve_path(args.train_json) if args.train_json else coco_dir / "annotations" / "captions_train2017.json"
    val_json = resolve_path(args.val_json) if args.val_json else coco_dir / "annotations" / "captions_val2017.json"
    train_image_dir = resolve_path(args.train_image_dir) if args.train_image_dir else coco_dir / "train2017"
    val_image_dir = resolve_path(args.val_image_dir) if args.val_image_dir else coco_dir / "val2017"

    output_train_csv = resolve_path(args.output_train_csv)
    output_val_csv = resolve_path(args.output_val_csv)
    if args.split in {"both", "train"}:
        train_rows = build_rows(
            train_json,
            train_image_dir,
            all_captions=args.all_captions,
            max_rows=args.max_train,
            seed=args.seed,
        )
        write_csv(output_train_csv, train_rows)
        print(f"Wrote {len(train_rows)} rows to {output_train_csv}")
    if args.split in {"both", "val"}:
        val_rows = build_rows(
            val_json,
            val_image_dir,
            all_captions=args.all_captions or args.all_val_captions,
            max_rows=args.max_val,
            seed=args.seed,
        )
        write_csv(output_val_csv, val_rows)
        print(f"Wrote {len(val_rows)} rows to {output_val_csv}")


if __name__ == "__main__":
    main()
