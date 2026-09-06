from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .schemas import TokenBatch


def _vision_sequence(features: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Normalise timm/HF feature layouts to B x tokens x channels."""
    if features.ndim == 2:
        return features.unsqueeze(1)
    if features.ndim == 3:
        if features.shape[-1] == expected_dim:
            return features
        if features.shape[1] == expected_dim:
            return features.transpose(1, 2)
    if features.ndim == 4:
        if features.shape[1] == expected_dim:  # BCHW
            return features.flatten(2).transpose(1, 2)
        if features.shape[-1] == expected_dim:  # BHWC
            return features.flatten(1, 2)
    raise ValueError(f"Unsupported vision feature layout {tuple(features.shape)} for dim={expected_dim}")


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class VisionTokenAdapter(nn.Module):
    def __init__(self, input_dim: int, bridge_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, bridge_dim))

    def forward(self, features: torch.Tensor, metadata: dict[str, Any] | None = None) -> TokenBatch:
        raw = _vision_sequence(features, self.input_dim)
        tokens = self.projection(raw)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return TokenBatch(tokens, mask, masked_mean(tokens, mask), metadata or {}).validate()


class TextTokenAdapter(nn.Module):
    def __init__(self, input_dim: int, bridge_dim: int = 256) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, bridge_dim))

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> TokenBatch:
        tokens = self.projection(features)
        mask = attention_mask.to(device=tokens.device, dtype=torch.bool)
        return TokenBatch(tokens, mask, masked_mean(tokens, mask), metadata or {}).validate()


class FrozenTokenExtractor(nn.Module):
    """Expose real token APIs from one restored Phase 1 alignment model."""

    def __init__(self, alignment_model: nn.Module, bridge_dim: int = 256) -> None:
        super().__init__()
        self.alignment_model = alignment_model
        for parameter in self.alignment_model.parameters():
            parameter.requires_grad = False
        self.alignment_model.eval()
        self.vision_adapter = VisionTokenAdapter(alignment_model.vision_encoder.output_dim, bridge_dim)
        self.text_adapter = TextTokenAdapter(alignment_model.text_encoder.output_dim, bridge_dim)

    def train(self, mode: bool = True) -> "FrozenTokenExtractor":
        super().train(mode)
        self.alignment_model.eval()
        return self

    def vision_tokens(self, images: torch.Tensor) -> TokenBatch:
        with torch.no_grad():
            payload = self.alignment_model.vision_encoder.forward_features(images)
        features = payload.get("local")
        if not isinstance(features, torch.Tensor):
            features = payload.get("global")
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("Selected vision encoder did not expose token/spatial features")
        return self.vision_adapter(features, {"encoder": self.alignment_model.vision_encoder.name})

    def text_tokens(self, captions: list[str]) -> TokenBatch:
        encoder = self.alignment_model.text_encoder
        prepared = encoder._prepare_captions(captions)
        if encoder.is_openclip:
            raise NotImplementedError("Selected Phase 2 text experts must expose Hugging Face last_hidden_state")
        tokenized = encoder.tokenizer(prepared, padding=True, truncation=True, return_tensors="pt").to(encoder.device)
        with torch.no_grad():
            outputs = encoder.encoder(**tokenized)
        return self.text_adapter(
            outputs.last_hidden_state,
            tokenized["attention_mask"],
            {"encoder": encoder.name},
        )

    def assert_frozen_backbones(self) -> None:
        if any(parameter.requires_grad for parameter in self.alignment_model.parameters()):
            raise RuntimeError("Phase 1 alignment/backbone parameters must remain frozen")

    def assert_no_backbone_gradients(self) -> None:
        if any(parameter.grad is not None for parameter in self.alignment_model.parameters()):
            raise RuntimeError("Frozen Phase 1 model received gradients")
