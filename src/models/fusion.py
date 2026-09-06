from __future__ import annotations

import torch
from torch import nn


class VisionFusionModule(nn.Module):
    """Fuse main VE features with optional BLF-local and BLF-global features."""

    def __init__(
        self,
        main_dim: int,
        local_dim: int = 128,
        global_dim: int = 128,
        output_dim: int = 512,
        use_local: bool = True,
        use_global: bool = True,
        fusion_type: str = "concat_mlp",
    ) -> None:
        super().__init__()
        self.use_local = use_local
        self.use_global = use_global
        self.fusion_type = fusion_type
        dims = [main_dim]
        if use_local:
            dims.append(local_dim)
        if use_global:
            dims.append(global_dim)
        if fusion_type == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(sum(dims), output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
            )
        elif fusion_type == "gated":
            self.projections = nn.ModuleList([nn.Linear(dim, output_dim) for dim in dims])
            self.gate_logits = nn.Parameter(torch.zeros(len(dims)))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        self.output_dim = output_dim

    def forward(
        self,
        main_feat: torch.Tensor,
        local_feat: torch.Tensor | None = None,
        global_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [main_feat]
        if self.use_local:
            if local_feat is None:
                raise ValueError("local_feat is required when use_local=True")
            features.append(local_feat)
        if self.use_global:
            if global_feat is None:
                raise ValueError("global_feat is required when use_global=True")
            features.append(global_feat)
        if self.fusion_type == "concat_mlp":
            return self.fusion(torch.cat(features, dim=-1))
        projected = torch.stack([proj(feat) for proj, feat in zip(self.projections, features, strict=True)], dim=1)
        weights = torch.softmax(self.gate_logits, dim=0).view(1, -1, 1)
        return (projected * weights).sum(dim=1)

