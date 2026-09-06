"""Fetch the Flickr30k Karpathy test split from the HF Hub mirror and write
it to data/flickr30k/ in the same (image_path, caption) CSV schema used by
data/val_all_captions.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "flickr30k"
IMAGE_DIR = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "test.csv"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    dataset = dataset.filter(lambda row: row["split"] == "test")

    rows: list[dict[str, str]] = []
    for record in dataset:
        image_id = record["img_id"]
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        if not image_path.is_file():
            record["image"].convert("RGB").save(image_path)
        relative_path = str(image_path.relative_to(ROOT))
        for caption in record["caption"]:
            rows.append({"image_path": relative_path, "caption": caption})

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    num_images = len({row["image_path"] for row in rows})
    print(f"wrote {len(rows)} caption rows across {num_images} images -> {OUT_CSV}")


if __name__ == "__main__":
    main()
