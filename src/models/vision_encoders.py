from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


VISION_MODEL_REGISTRY = {
    "efficientnet_b0": ("efficientnet_b0",),
    "mobilenetv3_large_100": ("mobilenetv3_large_100",),
    "convnext_tiny": ("convnext_tiny",),
    "deit_tiny_patch16_224": ("deit_tiny_patch16_224",),
    "convnextv2_tiny": (
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_tiny",
    ),
    "dinov2_vits14": (
        "vit_small_patch14_dinov2.lvd142m",
        "vit_small_patch14_dinov2",
    ),
    "swin_tiny": (
        "swin_tiny_patch4_window7_224.ms_in1k",
        "swin_tiny_patch4_window7_224",
    ),
}

HF_VISION_MODEL_REGISTRY = {
    "dinov2_vits14": "facebook/dinov2-small",
}


@dataclass(frozen=True)
class VisionEncoderInfo:
    name: str
    output_dim: int


def _candidate_model_names(name: str) -> tuple[str, ...]:
    model_names = VISION_MODEL_REGISTRY.get(name, (name,))
    if isinstance(model_names, str):
        return (model_names,)
    return tuple(model_names)


def _timm_create_kwargs(name: str) -> dict[str, Any]:
    if name == "dinov2_vits14":
        return {"img_size": 224}
    return {}


def _extract_local_features(features: Any) -> torch.Tensor | None:
    if isinstance(features, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "last_hidden_state", "features"):
            value = features.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in features.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                return value
        return None
    if isinstance(features, (list, tuple)):
        tensors = [feature for feature in features if isinstance(feature, torch.Tensor)]
        return tensors[-1] if tensors else None
    return features if isinstance(features, torch.Tensor) and features.ndim >= 3 else None


class TimmVisionBackbone(nn.Module):
    def __init__(self, model: nn.Module, model_name: str) -> None:
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | str | None]:
        features = self.model.forward_features(images) if hasattr(self.model, "forward_features") else None
        return {
            "global": self.forward(images),
            "local": _extract_local_features(features),
            "model_name": self.model_name,
        }


class HFDinoVisionBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel, Dinov2Config, Dinov2Model
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("transformers is required for Hugging Face vision encoders.") from exc
        self.model_name = model_name
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        elif model_name == "facebook/dinov2-small":
            # Keep saved DINOv2-S/14 checkpoints self-contained: the architecture
            # is known and only the checkpoint supplies weights.
            self.model = Dinov2Model(
                Dinov2Config(
                    hidden_size=384,
                    num_hidden_layers=12,
                    num_attention_heads=6,
                    mlp_ratio=4,
                    image_size=518,
                    patch_size=14,
                )
            )
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.output_dim = int(self.model.config.hidden_size)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | str]:
        outputs = self.model(pixel_values=images)
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1:] if tokens.shape[1] > 1 else tokens
        cls_token = tokens[:, 0]
        return {
            "global": cls_token if cls_token is not None else patch_tokens.mean(dim=1),
            "local": patch_tokens,
            "model_name": self.model_name,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)["global"]  # type: ignore[return-value]


class VisionEncoder(nn.Module):
    """Pretrained vision encoder wrapper with a small registry."""

    def __init__(self, name: str, pretrained: bool = True, freeze: bool = True, backend: str = "auto") -> None:
        super().__init__()
        if backend not in {"auto", "timm", "hf"}:
            raise ValueError("vision backend must be auto, timm, or hf")
        self.name = name
        self.freeze = freeze
        self.backend = backend
        if name == "openclip_vit_b32":
            if backend not in {"auto", "timm"}:
                raise ValueError("openclip_vit_b32 does not support the Hugging Face backend")
            self.encoder, self.output_dim = self._build_openclip(pretrained=pretrained)
        elif name in HF_VISION_MODEL_REGISTRY:
            if backend == "hf":
                self.encoder, self.output_dim = self._build_hf(name=name, pretrained=pretrained)
            elif backend == "timm":
                self.encoder, self.output_dim = self._build_timm(name=name, pretrained=pretrained)
            else:
                self.encoder, self.output_dim = self._build_timm_or_hf(name=name, pretrained=pretrained)
        else:
            if backend == "hf":
                raise ValueError(f"Vision encoder {name!r} does not have a Hugging Face backend")
            self.encoder, self.output_dim = self._build_timm(name=name, pretrained=pretrained)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.encoder(images)
        return self.encoder(images)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | str | None]:
        if self.freeze:
            with torch.no_grad():
                return self._forward_features(images)
        return self._forward_features(images)

    def _forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor | str | None]:
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(images)
            if isinstance(features, dict) and "global" in features:
                return features
        else:
            features = None
        return {
            "global": self.encoder(images),
            "local": _extract_local_features(features),
            "model_name": self.name,
        }

    def train(self, mode: bool = True) -> "VisionEncoder":
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    @staticmethod
    def _build_timm(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        try:
            import timm
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("timm is required for timm vision encoders. Install `timm`.") from exc
        errors: list[str] = []
        for model_name in _candidate_model_names(name):
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                    **_timm_create_kwargs(name),
                )
                output_dim = int(getattr(model, "num_features", 0))
                if output_dim <= 0:
                    raise RuntimeError(f"Could not infer output dimension for timm model {model_name}")
                return TimmVisionBackbone(model, model_name), output_dim
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        raise RuntimeError(f"Could not build timm vision encoder {name}. Tried: {'; '.join(errors)}")

    @staticmethod
    def _build_timm_or_hf(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        try:
            return VisionEncoder._build_timm(name=name, pretrained=pretrained)
        except Exception as timm_exc:
            try:
                return VisionEncoder._build_hf(name=name, pretrained=pretrained)
            except Exception as hf_exc:
                raise RuntimeError(
                    f"Could not build vision encoder {name}. "
                    f"timm failed with: {timm_exc}. Hugging Face fallback failed with: {hf_exc}"
                ) from hf_exc

    @staticmethod
    def _build_hf(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        model = HFDinoVisionBackbone(HF_VISION_MODEL_REGISTRY[name], pretrained=pretrained)
        return model, model.output_dim

    @staticmethod
    def _build_openclip(pretrained: bool) -> tuple[nn.Module, int]:
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("open_clip_torch is required for openclip_vit_b32.") from exc
        weights = "openai" if pretrained else None
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=weights)

        class OpenCLIPImageEncoder(nn.Module):
            def __init__(self, clip_model: nn.Module) -> None:
                super().__init__()
                self.clip_model = clip_model

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.clip_model.encode_image(images)

        output_dim = int(getattr(model.visual, "output_dim", 512))
        return OpenCLIPImageEncoder(model), output_dim
