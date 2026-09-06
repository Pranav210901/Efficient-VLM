from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.collate import image_text_collate


REFERENCE_CHECKPOINTS = {
    "openclip_vit_b32_quickgelu_openai": ("ViT-B-32-quickgelu", "openai"),
    "mobileclip2_s0_dfndr2b": ("MobileCLIP2-S0", "dfndr2b"),
    "siglip2_vit_b32_256_webli": ("ViT-B-32-SigLIP2-256", "webli"),
}


class NativePairedModel(nn.Module):
    """OpenCLIP-compatible paired checkpoint plus its checkpoint-native transform."""

    def __init__(self, model_name: str, pretrained: str) -> None:
        super().__init__()
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("open_clip_torch is required for paired references") from exc
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def embedding_dim(self) -> int:
        projection = getattr(self.model, "text_projection", None)
        if isinstance(projection, torch.Tensor):
            return int(projection.shape[-1])
        config = getattr(self.model, "embed_dim", None)
        if config:
            return int(config)
        with torch.no_grad():
            return int(self.encode_text(["test"]).shape[-1])

    @property
    def image_size(self) -> int:
        value = getattr(self.model.visual, "image_size", 224)
        if isinstance(value, (tuple, list)):
            return int(value[0])
        return int(value)

    @property
    def logit_scale_value(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        if isinstance(scale, torch.Tensor):
            return float(scale.detach().exp().clamp(max=100).cpu())
        return 1.0

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        return self.encode_text_tokens(self.tokenize(captions))

    @property
    def native_context_length(self) -> int:
        value = getattr(self.model, "context_length", None)
        if value is None:
            value = getattr(self.tokenizer, "context_length", 77)
        return int(value)

    def tokenize(self, captions: list[str], *, pad_to_native_context: bool = True) -> torch.Tensor:
        # OpenCLIP tokenizers emit their checkpoint-native fixed context.
        return self.tokenizer(captions)

    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_text(tokens.to(self.device)), dim=-1)

    def forward(self, images: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        image = self.encode_image(images)
        text = self.encode_text(captions)
        scale = torch.as_tensor(self.logit_scale_value, device=image.device, dtype=image.dtype)
        return {
            "image_embeds": image,
            "text_embeds": text,
            "logits": scale * image @ text.t(),
            "logit_scale": scale,
        }


class NativeGroupedDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        image_root: str | Path = ".",
        captions_per_image: int | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.captions_per_image = captions_per_image
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image_path", "caption"}.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain image_path and caption")
            for row in reader:
                grouped.setdefault(str(row["image_path"]), []).append(str(row["caption"]))
        self.samples = list(grouped.items())
        if not self.samples:
            raise ValueError(f"{csv_path} is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        value, captions = self.samples[index]
        path = Path(value)
        if not path.is_absolute():
            path = self.image_root / path
        selected = captions if self.captions_per_image is None else captions[: self.captions_per_image]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "caption": selected, "image_path": str(path)}


def native_loader(
    model: NativePairedModel,
    csv_path: str | Path,
    *,
    image_root: str | Path = ".",
    batch_size: int = 256,
    num_workers: int = 10,
    captions_per_image: int | None = None,
) -> DataLoader:
    dataset = NativeGroupedDataset(
        csv_path,
        model.preprocess,
        image_root=image_root,
        captions_per_image=captions_per_image,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=image_text_collate,
    )


def build_reference(reference_id: str) -> NativePairedModel:
    if reference_id not in REFERENCE_CHECKPOINTS:
        raise ValueError(f"unknown reference {reference_id!r}")
    return NativePairedModel(*REFERENCE_CHECKPOINTS[reference_id])


def preprocessing_description(model: NativePairedModel) -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "pretrained": model.pretrained,
        "image_size": model.image_size,
        "transform": repr(model.preprocess),
    }
