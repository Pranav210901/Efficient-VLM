from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.collate import image_text_collate


class ProjectorDriftTracker:
    # fixed-probe projector drift with frozen encoder features cached once

    def __init__(
        self,
        model: torch.nn.Module,
        dev_loader: DataLoader,
        *,
        manifest_path: str | Path,
        seed: int = 20260727,
        size: int = 512,
        interval: int = 10,
    ) -> None:
        dataset = dev_loader.dataset
        if len(dataset) < size:
            raise ValueError(f"DEV probe requires {size} unique images; found {len(dataset)}")
        indices = sorted(random.Random(seed).sample(range(len(dataset)), size))
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=size,
            shuffle=False,
            num_workers=0,
            collate_fn=image_text_collate,
        )
        batch = next(iter(loader))
        images = batch["images"].to(next(model.parameters()).device)
        captions = [str(value) for value in batch["captions"]]
        image_ids = [str(value) for value in batch["image_paths"]]
        text_ids = [str(value) for value in batch.get("text_image_paths", image_ids)]
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "caption"])
            writer.writeheader()
            writer.writerows(
                {"image_path": image_id, "caption": caption}
                for image_id, caption in zip(text_ids, captions)
            )
        self.interval = int(interval)
        self.model = model
        with torch.no_grad():
            image_features = model.vision_encoder.forward_features(images).global_feature
            text_features = model.text_encoder(captions)
        self.image_features = image_features.detach()
        self.text_features = text_features.detach()
        initial = self._project()
        self.initial = initial
        self.previous = initial

    def _project(self) -> tuple[torch.Tensor, torch.Tensor]:
        image_mode = self.model.image_projection.training
        text_mode = self.model.text_projection.training
        self.model.image_projection.eval()
        self.model.text_projection.eval()
        try:
            with torch.no_grad():
                image = F.normalize(
                    self.model.image_projection(self.image_features), dim=-1
                )
                text = F.normalize(
                    self.model.text_projection(self.text_features), dim=-1
                )
        finally:
            self.model.image_projection.train(image_mode)
            self.model.text_projection.train(text_mode)
        return image.detach(), text.detach()

    @staticmethod
    def _distance(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((1.0 - (left * right).sum(dim=-1)).mean().cpu())

    def measure(self, optimizer_step: int) -> dict[str, float] | None:
        if optimizer_step % self.interval:
            return None
        current = self._project()
        row = {
            "optimizer_step": int(optimizer_step),
            "image_from_step0": self._distance(current[0], self.initial[0]),
            "image_from_previous": self._distance(current[0], self.previous[0]),
            "text_from_step0": self._distance(current[1], self.initial[1]),
            "text_from_previous": self._distance(current[1], self.previous[1]),
        }
        self.previous = current
        return row

    def state_dict(self) -> dict[str, Any]:
        return {"initial": self.initial, "previous": self.previous}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            self.initial = tuple(state["initial"])
            self.previous = tuple(state["previous"])


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
