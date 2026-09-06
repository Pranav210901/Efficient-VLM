from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .collate import image_text_collate
from .transforms import build_image_transform


class ImageTextCSVDataset(Dataset[dict[str, object]]):
    """Generic image-text dataset loaded from a CSV with image_path,caption columns."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path = ".",
        image_size: int = 224,
        train: bool = True,
        *,
        group_by_image: bool = False,
        captions_per_image: int | None = None,
        image_mean: list[float] | tuple[float, ...] | None = None,
        image_std: list[float] | tuple[float, ...] | None = None,
        interpolation: str = "bicubic",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.train = bool(train)
        self.group_by_image = bool(group_by_image)
        self.captions_per_image = captions_per_image
        if captions_per_image is not None and int(captions_per_image) < 1:
            raise ValueError("captions_per_image must be positive or null")
        self.transform = build_image_transform(
            image_size=image_size,
            train=train,
            mean=image_mean,
            std=image_std,
            interpolation=interpolation,
        )
        rows = self._read_csv()
        self.samples = self._group_rows(rows) if self.group_by_image else rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        captions = sample["captions"] if "captions" in sample else sample["caption"]
        if isinstance(captions, list) and self.captions_per_image is not None and len(captions) > self.captions_per_image:
            if self.train:
                captions = random.sample(captions, self.captions_per_image)
            else:
                captions = captions[: self.captions_per_image]
        return {
            "image": self.transform(image),
            "caption": captions,
            "image_path": str(image_path),
        }

    @staticmethod
    def _group_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        for row in rows:
            grouped.setdefault(row["image_path"], []).append(row["caption"])
        return [{"image_path": image_path, "captions": captions} for image_path, captions in grouped.items()]

    def _read_csv(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}\n"
                "Prepare an image-caption dataset before training. For COCO Captions, run:\n"
                "  bash scripts/download_coco.sh\n"
                "or, if COCO is already downloaded under data/coco, run:\n"
                "  python scripts/prepare_coco.py\n"
                "For a quick smoke test after downloading COCO, run:\n"
                "  python scripts/prepare_coco.py --max_train 2000 --max_val 500"
            )
        with self.csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "caption"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"{self.csv_path} must contain columns: image_path, caption")
            rows = [{"image_path": str(row["image_path"]), "caption": str(row["caption"])} for row in reader]
        if not rows:
            raise ValueError(f"{self.csv_path} contained no samples")
        return rows


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    group_by_image = bool(data_cfg.get("group_by_image", False))
    common = {
        "image_root": data_cfg.get("image_root", "."),
        "image_size": int(data_cfg.get("image_size", 224)),
        "group_by_image": group_by_image,
        "image_mean": data_cfg.get("image_mean"),
        "image_std": data_cfg.get("image_std"),
        "interpolation": str(data_cfg.get("interpolation", "bicubic")),
    }
    train_dataset = ImageTextCSVDataset(
        csv_path=data_cfg["train_csv"],
        train=True,
        captions_per_image=data_cfg.get("train_captions_per_image") if group_by_image else None,
        **common,
    )
    val_dataset = ImageTextCSVDataset(
        csv_path=data_cfg["val_csv"],
        train=False,
        captions_per_image=data_cfg.get("val_captions_per_image") if group_by_image else None,
        **common,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    loader_tuning = {}
    if num_workers > 0:
        loader_tuning["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=image_text_collate,
        persistent_workers=bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
        drop_last=bool(config["training"].get("drop_last", False)),
        **loader_tuning,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=image_text_collate,
        persistent_workers=bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
        **loader_tuning,
    )
    return train_loader, val_loader
