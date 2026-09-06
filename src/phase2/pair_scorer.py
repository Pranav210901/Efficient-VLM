from __future__ import annotations

import torch
from torch import nn


class PairScorer(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
