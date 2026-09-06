from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from src.alignment_v3.model import (
    AlignmentV3Model,
    LoRALinear,
    build_model as build_alignment_model,
    install_text_lora,
)


def install_vision_lora(
    model: AlignmentV3Model,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    final_layers: int,
) -> list[str]:
    blocks = getattr(model.vision_encoder.model, "blocks", None)
    if blocks is None or len(blocks) < final_layers:
        raise ValueError("DINOv3 model does not expose enough transformer blocks")
    touched: list[str] = []
    for layer_index in range(len(blocks) - final_layers, len(blocks)):
        attention = blocks[layer_index].attn
        for projection_name in ("qkv", "proj"):
            projection = getattr(attention, projection_name)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"vision target blocks.{layer_index}.attn.{projection_name} "
                    "is not nn.Linear"
                )
            setattr(
                attention,
                projection_name,
                LoRALinear(projection, rank, alpha, dropout),
            )
            touched.append(f"blocks.{layer_index}.attn.{projection_name}")
    # Base weights remain requires_grad=False; gradients traverse them only to
    # reach the injected low-rank matrices.
    model.vision_encoder.freeze = False
    return touched


def build_model(config: dict[str, Any], *, pretrained: bool = True) -> AlignmentV3Model:
    adjusted = deepcopy(config)
    recipe = adjusted.setdefault("recipe", {})
    # FreezeShift owns both tower-specific switches. Disable the legacy
    # text-only switch to avoid double injection.
    recipe["lora"] = False
    model = build_alignment_model(adjusted, pretrained=pretrained)
    rank = int(recipe.get("freezeshift_lora_rank", 128))
    alpha = float(recipe.get("freezeshift_lora_alpha", 256.0))
    dropout = float(recipe.get("freezeshift_lora_dropout", 0.05))
    final_layers = int(recipe.get("freezeshift_lora_final_layers", 4))
    targets: list[str] = []
    if bool(recipe.get("freezeshift_vision_lora", False)):
        targets.extend(
            "vision." + target
            for target in install_vision_lora(
                model,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                final_layers=final_layers,
            )
        )
    if bool(recipe.get("freezeshift_text_lora", False)):
        targets.extend(
            "text." + target
            for target in install_text_lora(
                model.text_encoder,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                final_layers=final_layers,
            )
        )
    model.lora_targets = targets
    return model


def _merged_linear(module: LoRALinear) -> nn.Linear:
    base = module.base
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    delta = module.lora_b.weight @ module.lora_a.weight
    with torch.no_grad():
        merged.weight.copy_(base.weight + module.scaling * delta.to(base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias)
    merged.requires_grad_(False)
    return merged


def merge_lora_(module: nn.Module) -> list[str]:
    """Merge every LoRA branch into its base linear weight in-place."""
    merged: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                setattr(parent, name, _merged_linear(child))
                merged.append(path)
            else:
                visit(child, path)

    visit(module, "")
    return merged


def lora_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(part in {"lora_a", "lora_b"} for part in name.split("."))
    )
