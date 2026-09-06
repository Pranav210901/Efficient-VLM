from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    INVALID = "INVALID"


@dataclass
class TokenBatch:
    """Mask-aware token sequence exposed by a frozen pretrained encoder."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TokenBatch":
        if self.tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,D], got {tuple(self.tokens.shape)}")
        if self.attention_mask.shape != self.tokens.shape[:2]:
            raise ValueError("attention_mask must have shape [B,N]")
        if not self.tokens.is_floating_point():
            raise ValueError("tokens must use a floating-point dtype")
        if self.attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must use torch.bool")
        if self.attention_mask.device != self.tokens.device:
            raise ValueError("tokens and attention_mask must be on the same device")
        if self.pooled is not None and self.pooled.shape != (self.tokens.shape[0], self.tokens.shape[2]):
            raise ValueError("pooled must have shape [B,D]")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("token batch contains non-finite values")
        if not bool(self.attention_mask.any(dim=1).all()):
            raise ValueError("every sequence must contain at least one unmasked token")
        return self


REQUIRED_RUN_FILES = (
    "run_config.yaml",
    "environment.json",
    "status.json",
    "metrics_by_epoch.csv",
    "summary.json",
)
