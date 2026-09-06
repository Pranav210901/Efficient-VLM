from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .blf_branches import BLFGlobalBranch, BLFLocalBranch
from .fusion import VisionFusionModule
from .projection import build_projection_head
from .text_encoders import TextEncoder
from .vision_encoders import VisionEncoder


class BLFVisionLanguageModel(nn.Module):
    def __init__(
        self,
        vision_encoder_name: str,
        text_encoder_name: str,
        use_local_blf: bool,
        use_global_blf: bool,
        fusion_type: str,
        shared_dim: int = 256,
        local_dim: int = 128,
        global_dim: int = 128,
        fusion_dim: int = 512,
        projection_hidden_dim: int = 512,
        projection_dropout: float = 0.1,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        pretrained_vision: bool = True,
        pretrained_text: bool = True,
        vision_backend: str = "auto",
        text_prefix: str | None = None,
        projection_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(
            vision_encoder_name,
            pretrained=pretrained_vision,
            freeze=freeze_vision,
            backend=vision_backend,
        )
        self.text_encoder = TextEncoder(text_encoder_name, pretrained=pretrained_text, freeze=freeze_text, text_prefix=text_prefix)
        self.use_local_blf = use_local_blf
        self.use_global_blf = use_global_blf
        self.local_branch = BLFLocalBranch(output_dim=local_dim) if use_local_blf else None
        self.global_branch = BLFGlobalBranch(output_dim=global_dim) if use_global_blf else None
        self.vision_fusion = VisionFusionModule(
            main_dim=self.vision_encoder.output_dim,
            local_dim=local_dim,
            global_dim=global_dim,
            output_dim=fusion_dim,
            use_local=use_local_blf,
            use_global=use_global_blf,
            fusion_type=fusion_type,
        )
        self.image_projection = build_projection_head(
            projection_type, fusion_dim, shared_dim, projection_hidden_dim, projection_dropout
        )
        self.text_projection = build_projection_head(
            projection_type, self.text_encoder.output_dim, shared_dim, projection_hidden_dim, projection_dropout
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        main_feat = self.vision_encoder(images)
        local_feat = self.local_branch(images) if self.local_branch is not None else None
        global_feat = self.global_branch(images) if self.global_branch is not None else None
        fused = self.vision_fusion(main_feat, local_feat=local_feat, global_feat=global_feat)
        return F.normalize(self.image_projection(fused), dim=-1)

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        text_feat = self.text_encoder(captions)
        return F.normalize(self.text_projection(text_feat), dim=-1)

    def forward(self, images: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(captions)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * image_embeds @ text_embeds.t()
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits": logits,
            "logit_scale": logit_scale,
        }
