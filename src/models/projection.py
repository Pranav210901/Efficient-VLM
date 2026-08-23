from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualProjectionHead(nn.Module):
    # low-capacity residual adapter suited to frozen encoder features

    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x)
        return self.skip(normalised) + self.residual_scale.sigmoid() * self.residual(normalised)


def build_projection_head(
    projection_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if projection_type == "mlp":
        return ProjectionHead(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    if projection_type == "residual":
        return ResidualProjectionHead(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
    if projection_type == "linear":
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim, bias=False))
    raise ValueError(f"unknown projection_type {projection_type!r}")
