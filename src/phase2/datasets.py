from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _prompt(value: object) -> str:
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return str(value)


class RetrievalHardNegativeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable, negative_source: str = "hard") -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.negative_source = negative_source

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        negative_row = self.frame.iloc[(index * 7919 + 104729) % len(self.frame)] if self.negative_source == "random" else row
        if self.negative_source == "random":
            offset = 0
            while str(negative_row.positive_id) == str(row.positive_id) and offset < len(self.frame):
                offset += 1; negative_row = self.frame.iloc[(index * 7919 + 104729 + offset) % len(self.frame)]
            if str(negative_row.positive_id) == str(row.positive_id):
                raise RuntimeError("Random-negative pool contains no distinct image")
        positive_image = str(row.positive_id)
        if str(row.direction) == "i2t":
            negative_image = positive_image
            positive_caption = str(row.positive_text)
            negative_caption = str(negative_row.positive_text) if self.negative_source == "random" else str(row.negative_text)
        else:
            negative_image = str(negative_row.positive_id) if self.negative_source == "random" else str(row.negative_id)
            positive_caption = negative_caption = str(row.positive_text)
        return {
            "positive_image": self.transform(_rgb(positive_image)),
            "positive_caption": positive_caption,
            "negative_image": self.transform(_rgb(negative_image)),
            "negative_caption": negative_caption,
            "query_id": str(row.query_id),
        }


class ClassificationHardNegativeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_resolver: Callable[[pd.Series], Image.Image], transform: Callable) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_resolver = image_resolver
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        resolved = self.image_resolver(row)
        image = resolved if isinstance(resolved, torch.Tensor) else self.transform(resolved.convert("RGB"))
        return {
            "positive_image": image,
            "negative_image": image.clone(),
            "positive_caption": _prompt(row.true_prompts),
            "negative_caption": _prompt(row.negative_prompts),
            "query_id": str(row.sample_id),
        }


def bridge_collate(rows: list[dict]) -> dict[str, object]:
    return {
        "positive_images": torch.stack([row["positive_image"] for row in rows]),
        "negative_images": torch.stack([row["negative_image"] for row in rows]),
        "positive_captions": [row["positive_caption"] for row in rows],
        "negative_captions": [row["negative_caption"] for row in rows],
        "query_ids": [row["query_id"] for row in rows],
    }
