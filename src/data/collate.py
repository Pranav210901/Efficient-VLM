from __future__ import annotations

import torch


def image_text_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    captions: list[str] = []
    text_image_paths: list[str] = []
    for item in batch:
        image_path = str(item["image_path"])
        value = item["caption"]
        item_captions = [str(caption) for caption in value] if isinstance(value, (list, tuple)) else [str(value)]
        captions.extend(item_captions)
        text_image_paths.extend([image_path] * len(item_captions))
    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "captions": captions,
        "image_paths": [str(item["image_path"]) for item in batch],
        "text_image_paths": text_image_paths,
    }
