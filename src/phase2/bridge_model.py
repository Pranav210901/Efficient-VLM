from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.multitask.checkpoint_selection import CheckpointSpec, load_frozen_model

from .cross_attention import TextToVisionBridge
from .pair_scorer import PairScorer
from .token_adapters import FrozenTokenExtractor
from .schemas import TokenBatch


class FrozenPairBridge(nn.Module):
    """Trainable token adapters/bridge over one entirely frozen Phase 1 pair."""

    def __init__(
        self,
        alignment_model: nn.Module,
        bridge_dim: int = 256,
        attention_heads: int = 4,
        layers: int = 1,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "masked_mean",
    ) -> None:
        super().__init__()
        self.tokens = FrozenTokenExtractor(alignment_model, bridge_dim)
        self.bridge = TextToVisionBridge(bridge_dim, attention_heads, layers, feedforward_dim, dropout, pooling)
        self.scorer = PairScorer(bridge_dim, bridge_dim, dropout)
        self.tokens.assert_frozen_backbones()

    def forward(self, images: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        vision = self.tokens.vision_tokens(images)
        text = self.tokens.text_tokens(captions)
        return self.forward_tokens(vision, text)

    def forward_tokens(self, vision: TokenBatch, text: TokenBatch) -> dict[str, torch.Tensor]:
        features = self.bridge(text, vision)
        scores = self.scorer(features)
        return {"scores": scores, "features": features, "vision_tokens": vision.tokens, "text_tokens": text.tokens}

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def validate_gradients(self) -> None:
        self.tokens.assert_no_backbone_gradients()
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable or not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("No bridge/adapter gradient was produced")

    @classmethod
    def from_selected_pair(cls, pair: dict[str, Any], device: str | torch.device = "cpu", **kwargs: Any) -> "FrozenPairBridge":
        spec = CheckpointSpec(
            str(pair["vision_encoder"]),
            str(pair["text_encoder"]),
            str(pair.get("variant", "baseline")),
            Path(pair["checkpoint_path"]),
            0,
        )
        model, _, _ = load_frozen_model(spec, str(device))
        return cls(model, **kwargs).to(device)
