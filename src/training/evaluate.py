from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .losses import multi_positive_contrastive_loss
from .metrics import compute_retrieval_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    image_paths: list[str] = []
    text_image_paths: list[str] = []
    for batch in loader:
        images = batch["images"].to(device)
        captions = list(batch["captions"])
        outputs: dict[str, Any] = model(images, captions)
        batch_image_paths = [str(path) for path in batch["image_paths"]]
        batch_text_image_paths = [str(path) for path in batch.get("text_image_paths", batch["image_paths"])]
        losses.append(
            float(
                multi_positive_contrastive_loss(
                    outputs["logits"],
                    batch_image_paths,
                    batch_text_image_paths,
                ).item()
            )
        )
        image_parts.append(outputs["image_embeds"].detach().cpu())
        text_parts.append(outputs["text_embeds"].detach().cpu())
        image_paths.extend(batch_image_paths)
        text_image_paths.extend(batch_text_image_paths)
    image_embeds = torch.cat(image_parts, dim=0)
    text_embeds = torch.cat(text_parts, dim=0)
    metrics = compute_retrieval_metrics(
        image_embeds,
        text_embeds,
        k_values=k_values,
        image_ids=image_paths,
        text_image_ids=text_image_paths,
    )
    # Diagonal CLIP loss is not meaningful when several captions share the same image,
    # because valid matches would be counted as negatives. Retrieval metrics above
    # remain multi-positive aware.
    metrics["val_loss"] = sum(losses) / max(1, len(losses))
    metrics["mean_R@1"] = 0.5 * (metrics["i2t_R@1"] + metrics["t2i_R@1"])
    return metrics


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    paths: list[str] = []
    captions: list[str] = []
    text_image_paths: list[str] = []
    for batch in loader:
        images = batch["images"].to(device)
        batch_captions = list(batch["captions"])
        image_parts.append(model.encode_image(images).detach().cpu())
        text_parts.append(model.encode_text(batch_captions).detach().cpu())
        paths.extend([str(path) for path in batch["image_paths"]])
        text_image_paths.extend([str(path) for path in batch.get("text_image_paths", batch["image_paths"])])
        captions.extend(batch_captions)
    return {
        "image_embeds": torch.cat(image_parts, dim=0),
        "text_embeds": torch.cat(text_parts, dim=0),
        "image_paths": paths,
        "captions": captions,
        "text_image_paths": text_image_paths,
    }
