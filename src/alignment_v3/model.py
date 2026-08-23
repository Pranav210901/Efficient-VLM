from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.projection import ResidualProjectionHead
from src.models.text_encoders import TextEncoder, mean_pool


DINO_VISION_MODELS = {
    "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
}


@dataclass
class VisionFeatures:
    global_feature: torch.Tensor
    local_tokens: torch.Tensor
    spatial_shape: tuple[int, int]


class DINOv3VisionEncoder(nn.Module):
    # frozen timm DINOv3 encoder exposing one global vector and spatial tokens

    def __init__(self, name: str, *, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        if name not in DINO_VISION_MODELS:
            raise ValueError(f"unknown DINOv3 encoder {name!r}; expected one of {sorted(DINO_VISION_MODELS)}")
        try:
            import timm
        except Exception as exc:  # pragma: no cover - dependency validation covers this
            raise RuntimeError("timm is required for DINOv3 encoders") from exc
        self.name = name
        self.checkpoint_id = DINO_VISION_MODELS[name]
        self.model = timm.create_model(self.checkpoint_id, pretrained=pretrained, num_classes=0)
        self.output_dim = int(self.model.num_features)
        self.freeze = bool(freeze)
        if self.freeze:
            self.requires_grad_(False)
            self.eval()

    def _features(self, images: torch.Tensor) -> VisionFeatures:
        raw = self.model.forward_features(images)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"{self.checkpoint_id}.forward_features returned {type(raw).__name__}")
        if raw.ndim == 4:
            batch, channels, height, width = raw.shape
            local = raw.flatten(2).transpose(1, 2).contiguous()
            global_feature = self.model.forward_head(raw, pre_logits=True)
            if global_feature.ndim > 2:
                global_feature = global_feature.mean(dim=tuple(range(2, global_feature.ndim)))
            if local.shape != (batch, height * width, channels):
                raise AssertionError("ConvNeXt spatial flattening produced an invalid shape")
            return VisionFeatures(global_feature, local, (height, width))
        if raw.ndim == 3:
            prefix = int(getattr(self.model, "num_prefix_tokens", 1))
            if prefix < 1 or raw.shape[1] <= prefix:
                raise ValueError(f"invalid prefix-token count {prefix} for shape {tuple(raw.shape)}")
            local = raw[:, prefix:].contiguous()
            token_count = local.shape[1]
            side = int(round(math.sqrt(token_count)))
            if side * side != token_count:
                raise ValueError(f"ViT patch-token count {token_count} is not a square grid")
            return VisionFeatures(raw[:, 0], local, (side, side))
        raise ValueError(f"unsupported DINOv3 feature shape {tuple(raw.shape)}")

    def forward_features(self, images: torch.Tensor) -> VisionFeatures:
        if self.freeze:
            with torch.no_grad():
                return self._features(images)
        return self._features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images).global_feature

    def train(self, mode: bool = True) -> "DINOv3VisionEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self


def sinusoidal_2d_position(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D sinusoidal position dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    position = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return position.to(dtype=dtype).unsqueeze(0)


class SpatialTokenAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_dim: int,
        *,
        adapter_dim: int = 128,
        heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        blocks: int = 1,
        native_width_identity: bool = False,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("token adapter blocks must be positive")
        if native_width_identity and adapter_dim != input_dim:
            raise ValueError("native-width identity requires adapter_dim == input_dim")
        self.input_projection = (
            nn.Identity()
            if native_width_identity
            else nn.Linear(input_dim, adapter_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=adapter_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Preserve the historical one-block state-dict layout exactly.
        self.encoder = (
            layer
            if blocks == 1
            else nn.TransformerEncoder(layer, num_layers=blocks)
        )
        self.output_projection = (
            nn.Identity()
            if native_width_identity and adapter_dim == global_dim
            else nn.Linear(adapter_dim, global_dim)
        )
        self.output_norm = nn.LayerNorm(global_dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_feature: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        height, width = spatial_shape
        if local_tokens.ndim != 3 or local_tokens.shape[1] != height * width:
            raise ValueError(
                f"local tokens {tuple(local_tokens.shape)} do not match spatial shape {spatial_shape}"
            )
        tokens = self.input_projection(local_tokens)
        tokens = tokens + sinusoidal_2d_position(
            height, width, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype
        )
        pooled = self.encoder(tokens).mean(dim=1)
        update = self.output_projection(pooled)
        return self.output_norm(global_feature + self.gate.sigmoid() * update)


class CLSMeanPatchFusion(nn.Module):
    # fuse CLS with the uniform mean of patch tokens, excluding registers

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_feature: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        if local_tokens.shape[1] != spatial_shape[0] * spatial_shape[1]:
            raise ValueError("patch-token count does not match spatial shape")
        return self.fusion(torch.cat((global_feature, local_tokens.mean(dim=1)), dim=-1))


# learned queries attend over the patch grid rather than mean-pooling it. this is the
# C4 aggregator and it carries most of the trainable budget
class LearnedQueryPatchPool(nn.Module):
    # one learned query cross-attending to patches; no patch-to-patch attention

    def __init__(self, input_dim: int = 384, pool_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if pool_dim % heads:
            raise ValueError("pool dimension must be divisible by attention heads")
        self.input_projection = nn.Linear(input_dim, pool_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, pool_dim))
        nn.init.normal_(self.query, std=pool_dim**-0.5)
        self.attention = nn.MultiheadAttention(
            pool_dim, heads, dropout=0.0, batch_first=True
        )
        self.output_projection = nn.Linear(pool_dim, input_dim)
        self.output_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_feature: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        if local_tokens.shape[1] != spatial_shape[0] * spatial_shape[1]:
            raise ValueError("patch-token count does not match spatial shape")
        tokens = self.input_projection(local_tokens)
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        update = self.output_projection(pooled[:, 0])
        return self.output_norm(global_feature + self.gate.sigmoid() * update)


class LearnedQueryTextPool(nn.Module):
    # learned-query pooling over valid frozen text tokens

    def __init__(self, input_dim: int = 384, pool_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if pool_dim % heads:
            raise ValueError("text pool dimension must be divisible by attention heads")
        self.input_projection = nn.Linear(input_dim, pool_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, pool_dim))
        nn.init.normal_(self.query, std=pool_dim**-0.5)
        self.attention = nn.MultiheadAttention(
            pool_dim, heads, dropout=0.0, batch_first=True
        )
        self.output_projection = nn.Linear(pool_dim, input_dim)
        self.output_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        if token_states.ndim != 3 or attention_mask.shape != token_states.shape[:2]:
            raise ValueError("text token states and attention mask do not align")
        valid = attention_mask.to(device=token_states.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("every caption must contain at least one valid token")
        values = self.input_projection(token_states)
        query = self.query.expand(values.shape[0], -1, -1)
        pooled, _ = self.attention(
            query,
            values,
            values,
            key_padding_mask=~valid,
            need_weights=False,
        )
        update = self.output_projection(pooled[:, 0])
        return self.output_norm(baseline + self.gate.sigmoid() * update)


# low-rank branch added beside a frozen linear; merge_lora_ folds it back at inference
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.scaling * self.lora_b(self.lora_a(self.dropout(values)))


def install_text_lora(
    text_encoder: TextEncoder,
    *,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    final_layers: int = 2,
) -> list[str]:
    base = text_encoder.encoder
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None or len(layers) < final_layers:
        raise ValueError(f"text model {type(base).__name__} does not expose BERT encoder.layer")
    touched: list[str] = []
    for layer_index in range(len(layers) - final_layers, len(layers)):
        attention = layers[layer_index].attention.self
        for projection_name in ("query", "value"):
            projection = getattr(attention, projection_name)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"target {layer_index}.{projection_name} is not nn.Linear")
            setattr(attention, projection_name, LoRALinear(projection, rank, alpha, dropout))
            touched.append(f"encoder.layer.{layer_index}.attention.self.{projection_name}")
    # Base weights remain frozen, but autograd must traverse the encoder to reach LoRA.
    text_encoder.freeze = False
    return touched


class AlignmentV3Model(nn.Module):
    def __init__(
        self,
        *,
        vision_encoder: str,
        text_encoder: str,
        shared_dim: int = 384,
        projection_hidden_dim: int = 768,
        projection_dropout: float = 0.05,
        text_prefix: str | None = None,
        use_adapter: bool = False,
        image_token_aggregation: str = "cls",
        text_token_aggregation: str = "masked_mean",
        text_pooler_dim: int = 128,
        text_pooler_heads: int = 4,
        token_pooler_dim: int = 128,
        token_pooler_heads: int = 4,
        token_pooler_blocks: int = 1,
        token_pooler_ff_dim: int = 256,
        token_pooler_native_width_identity: bool = False,
        use_lora: bool = False,
        use_distillation: bool = False,
        teacher_dim: int = 512,
        teacher_dims: dict[str, int] | None = None,
        adapter_dim: int = 128,
        adapter_heads: int = 4,
        adapter_ff_dim: int = 256,
        adapter_dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.05,
        lora_final_layers: int = 2,
        loss_type: str = "infonce_queue",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder = DINOv3VisionEncoder(vision_encoder, pretrained=pretrained, freeze=True)
        self.text_encoder = TextEncoder(
            text_encoder, pretrained=pretrained, freeze=True, text_prefix=text_prefix
        )
        self.use_adapter = bool(use_adapter)
        self.use_lora = bool(use_lora)
        self.use_distillation = bool(use_distillation)
        self.adapter = (
            SpatialTokenAdapter(
                self.vision_encoder.output_dim,
                self.vision_encoder.output_dim,
                adapter_dim=adapter_dim,
                heads=adapter_heads,
                ff_dim=adapter_ff_dim,
                dropout=adapter_dropout,
            )
            if self.use_adapter
            else None
        )
        self.image_token_aggregation = str(image_token_aggregation)
        if self.use_adapter and self.image_token_aggregation != "cls":
            raise ValueError(
                "generic adapter and image_token_aggregation cannot be enabled together"
            )
        if self.image_token_aggregation == "cls":
            self.token_aggregator = None
        elif self.image_token_aggregation == "cls_mean_concat":
            self.token_aggregator = CLSMeanPatchFusion(self.vision_encoder.output_dim)
        elif self.image_token_aggregation == "learned_query_attention":
            self.token_aggregator = LearnedQueryPatchPool(
                self.vision_encoder.output_dim,
                pool_dim=token_pooler_dim,
                heads=token_pooler_heads,
            )
        elif self.image_token_aggregation == "transformer_128":
            self.token_aggregator = SpatialTokenAdapter(
                self.vision_encoder.output_dim,
                self.vision_encoder.output_dim,
                adapter_dim=token_pooler_dim,
                heads=token_pooler_heads,
                ff_dim=token_pooler_ff_dim,
                dropout=projection_dropout,
                blocks=token_pooler_blocks,
                native_width_identity=token_pooler_native_width_identity,
            )
        else:
            raise ValueError(
                f"unknown image_token_aggregation {self.image_token_aggregation!r}"
            )
        self.text_token_aggregation = str(text_token_aggregation)
        if self.text_token_aggregation == "masked_mean":
            self.text_token_aggregator = None
        elif self.text_token_aggregation == "learned_query_attention":
            if self.text_encoder.is_openclip:
                raise ValueError("learned text-token aggregation requires token states")
            self.text_token_aggregator = LearnedQueryTextPool(
                self.text_encoder.output_dim,
                pool_dim=int(text_pooler_dim),
                heads=int(text_pooler_heads),
            )
        else:
            raise ValueError(
                f"unknown text_token_aggregation {self.text_token_aggregation!r}"
            )
        self.image_projection = ResidualProjectionHead(
            self.vision_encoder.output_dim,
            output_dim=shared_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
        )
        self.text_projection = ResidualProjectionHead(
            self.text_encoder.output_dim,
            output_dim=shared_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.loss_type = str(loss_type)
        if self.loss_type == "sigmoid":
            self.sigmoid_logit_scale = nn.Parameter(torch.ones([]) * math.log(10.0))
            self.sigmoid_logit_bias = nn.Parameter(torch.ones([]) * -10.0)
        else:
            self.register_parameter("sigmoid_logit_scale", None)
            self.register_parameter("sigmoid_logit_bias", None)
        self.lora_targets: list[str] = []
        if self.use_lora:
            self.lora_targets = install_text_lora(
                self.text_encoder,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                final_layers=lora_final_layers,
            )
        configured_teacher_dims = {
            str(name): int(dimension)
            for name, dimension in (teacher_dims or {}).items()
        }
        if self.use_distillation and configured_teacher_dims:
            if any(dimension <= 0 for dimension in configured_teacher_dims.values()):
                raise ValueError("every teacher embedding dimension must be positive")
            self.teacher_heads = nn.ModuleDict(
                {
                    name: nn.ModuleDict(
                        {
                            "image": nn.Linear(shared_dim, dimension, bias=False),
                            "text": nn.Linear(shared_dim, dimension, bias=False),
                        }
                    )
                    for name, dimension in configured_teacher_dims.items()
                }
            )
            self.teacher_image_head = None
            self.teacher_text_head = None
        else:
            self.teacher_heads = nn.ModuleDict()
            self.teacher_image_head = (
                nn.Linear(shared_dim, teacher_dim, bias=False)
                if self.use_distillation
                else None
            )
            self.teacher_text_head = (
                nn.Linear(shared_dim, teacher_dim, bias=False)
                if self.use_distillation
                else None
            )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder.forward_features(images)
        global_feature = features.global_feature
        if self.token_aggregator is not None:
            global_feature = self.token_aggregator(
                features.local_tokens, global_feature, features.spatial_shape
            )
        if self.adapter is not None:
            global_feature = self.adapter(features.local_tokens, global_feature, features.spatial_shape)
        return F.normalize(self.image_projection(global_feature), dim=-1)

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        return self.encode_text_tokens(self.text_encoder.tokenize(captions))

    def encode_text_tokens(self, tokens: object) -> torch.Tensor:
        if self.text_token_aggregator is None:
            pooled = self.text_encoder.forward_tokens(tokens)
        else:
            token_states, attention_mask = self.text_encoder.forward_token_features(tokens)
            baseline = mean_pool(token_states, attention_mask)
            pooled = self.text_token_aggregator(token_states, attention_mask, baseline)
        return F.normalize(self.text_projection(pooled), dim=-1)

    def forward(self, images: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(captions)
        scale = self.logit_scale.exp().clamp(max=100.0)
        output = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits": scale * image_embeds @ text_embeds.t(),
            "logit_scale": scale,
        }
        if self.sigmoid_logit_scale is not None and self.sigmoid_logit_bias is not None:
            output["sigmoid_logit_scale"] = self.sigmoid_logit_scale.exp()
            output["sigmoid_logit_bias"] = self.sigmoid_logit_bias
        if self.teacher_image_head is not None and self.teacher_text_head is not None:
            output["teacher_space_image"] = F.normalize(self.teacher_image_head(image_embeds), dim=-1)
            output["teacher_space_text"] = F.normalize(self.teacher_text_head(text_embeds), dim=-1)
        if self.teacher_heads:
            output["teacher_spaces"] = {
                name: {
                    "image": F.normalize(heads["image"](image_embeds), dim=-1),
                    "text": F.normalize(heads["text"](text_embeds), dim=-1),
                }
                for name, heads in self.teacher_heads.items()
            }
        return output

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith(
                ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
            )
        }

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        training_only = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(
                ("teacher_image_head.", "teacher_text_head.", "teacher_heads.")
            )
        )
        return {
            "params_total_training": total,
            "params_total_inference": total - training_only,
            "params_trainable_training": trainable,
            "params_trainable_inference": trainable - training_only,
            "params_training_only": training_only,
        }


def build_model(config: dict[str, Any], *, pretrained: bool = True) -> AlignmentV3Model:
    model_cfg = config["model"]
    recipe = config.get("recipe", {})
    return AlignmentV3Model(
        vision_encoder=str(model_cfg["vision_encoder"]),
        text_encoder=str(model_cfg["text_encoder"]),
        shared_dim=int(model_cfg.get("shared_dim", 384)),
        projection_hidden_dim=int(model_cfg.get("projection_hidden_dim", 768)),
        projection_dropout=float(model_cfg.get("projection_dropout", 0.05)),
        text_prefix=model_cfg.get("text_prefix"),
        use_adapter=bool(recipe.get("adapter", False)),
        image_token_aggregation=str(recipe.get("image_token_aggregation", "cls")),
        text_token_aggregation=str(recipe.get("text_token_aggregation", "masked_mean")),
        text_pooler_dim=int(recipe.get("text_pooler_dim", 128)),
        text_pooler_heads=int(recipe.get("text_pooler_heads", 4)),
        token_pooler_dim=int(recipe.get("token_pooler_dim", 128)),
        token_pooler_heads=int(recipe.get("token_pooler_heads", 4)),
        token_pooler_blocks=int(recipe.get("token_pooler_blocks", 1)),
        token_pooler_ff_dim=int(recipe.get("token_pooler_ff_dim", 256)),
        token_pooler_native_width_identity=bool(
            recipe.get("token_pooler_native_width_identity", False)
        ),
        use_lora=bool(recipe.get("lora", False)),
        use_distillation=bool(recipe.get("distillation", False)),
        teacher_dim=int(recipe.get("teacher_dim", 512)),
        teacher_dims={
            str(name): int(dimension)
            for name, dimension in recipe.get("teacher_dims", {}).items()
        },
        adapter_dim=int(recipe.get("adapter_dim", 128)),
        adapter_heads=int(recipe.get("adapter_heads", 4)),
        adapter_ff_dim=int(recipe.get("adapter_ff_dim", 256)),
        adapter_dropout=float(recipe.get("adapter_dropout", 0.1)),
        lora_rank=int(recipe.get("lora_rank", 8)),
        lora_alpha=float(recipe.get("lora_alpha", 16)),
        lora_dropout=float(recipe.get("lora_dropout", 0.05)),
        lora_final_layers=int(recipe.get("lora_final_layers", 2)),
        loss_type=str(recipe.get("loss_type", "infonce_queue")),
        pretrained=pretrained,
    )
