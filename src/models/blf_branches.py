from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BLFLocalBranch(nn.Module):
    """Lightweight local texture/detail branch."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            DepthwiseSeparableConv(32),
            DepthwiseSeparableConv(32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class BLFGlobalBranch(nn.Module):
    """Lightweight global context branch."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=16, stride=16, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

