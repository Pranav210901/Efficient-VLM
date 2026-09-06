from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairedOpenCLIPModel(nn.Module):
    """A checkpoint-native paired OpenCLIP model used as an external reference."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        super().__init__()
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover - exercised on GPU nodes
            raise RuntimeError(
                "open_clip_torch is required for the paired reference model; "
                "install the project requirements in the Slurm environment"
            ) from exc
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.eval()

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(captions).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)

    def forward(self, images: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(captions)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits": scale * image_embeds @ text_embeds.t(),
            "logit_scale": scale,
        }
