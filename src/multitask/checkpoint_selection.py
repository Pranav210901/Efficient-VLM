from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from dataclasses import replace

from .config import TEXT_ALIASES, canonical_text_encoder

VARIANTS = ("local_global", "baseline", "global", "local")


@dataclass(frozen=True)
class CheckpointSpec:
    vision_encoder: str
    text_encoder: str
    variant: str
    checkpoint: Path
    epoch: int

    @property
    def config_id(self) -> str:
        return f"{self.vision_encoder}__{canonical_text_encoder(self.text_encoder)}__{self.variant}"


def parse_checkpoint_dir(name: str, vision_encoders: tuple[str, ...] | list[str]) -> tuple[str, str, str] | None:
    variant = next((value for value in VARIANTS if name.endswith("_" + value)), None)
    if variant is None:
        return None
    stem = name[: -(len(variant) + 1)]
    vision = next((value for value in sorted(vision_encoders, key=len, reverse=True) if stem.startswith(value + "_")), None)
    return None if vision is None else (vision, stem[len(vision) + 1 :], variant)


def resolve_best_checkpoints(
    checkpoint_root: str | Path,
    vision_encoders: tuple[str, ...] | list[str],
    variants: tuple[str, ...] = ("baseline",),
    deduplicate_aliases: bool = True,
    config_ids: set[str] | None = None,
    load_epochs: bool = True,
) -> list[CheckpointSpec]:
    root = Path(checkpoint_root)
    found: dict[str, CheckpointSpec] = {}
    for path in sorted(root.glob("*/best.pt")):
        parsed = parse_checkpoint_dir(path.parent.name, vision_encoders)
        if parsed is None or parsed[2] not in variants:
            continue
        vision, text, variant = parsed
        spec = CheckpointSpec(vision, text, variant, path, 0)
        key = spec.config_id if deduplicate_aliases else f"{vision}__{text}__{variant}"
        if config_ids is not None and key not in config_ids:
            continue
        previous = found.get(key)
        # Prefer the canonical public name when aliases point at the same backbone.
        if previous is None or (previous.text_encoder in TEXT_ALIASES and text not in TEXT_ALIASES):
            found[key] = spec
    selected = sorted(found.values(), key=lambda value: value.config_id)
    if load_epochs:
        selected = [
            replace(spec, epoch=int(torch.load(spec.checkpoint, map_location="cpu", weights_only=False).get("epoch", 0)))
            for spec in selected
        ]
    return selected


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor], expected_keys: set[str]) -> dict[str, torch.Tensor]:
    """Adapt checkpoints saved before timm backbones gained a wrapper module.

    The model parameters are unchanged; only the registered module path moved
    from ``vision_encoder.encoder.*`` to ``vision_encoder.encoder.model.*``.
    Keys are remapped only when the exact candidate exists in the current model.
    """
    old_prefix = "vision_encoder.encoder."
    new_prefix = "vision_encoder.encoder.model."
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        candidate = new_prefix + key[len(old_prefix):] if key.startswith(old_prefix) and not key.startswith(new_prefix) else key
        remapped[candidate if candidate in expected_keys else key] = value
    return remapped


def infer_vision_backend(state_dict: dict[str, torch.Tensor]) -> str:
    """Infer the DINO wrapper used when a checkpoint was saved."""
    keys = state_dict.keys()
    if any(
        key.startswith("vision_encoder.encoder.model.embeddings.")
        or key.startswith("vision_encoder.encoder.embeddings.")
        for key in keys
    ):
        return "hf"
    if any(
        key.startswith("vision_encoder.encoder.model.patch_embed.")
        or key.startswith("vision_encoder.encoder.patch_embed.")
        or key == "vision_encoder.encoder.model.cls_token"
        or key == "vision_encoder.encoder.cls_token"
        for key in keys
    ):
        return "timm"
    return "auto"


def load_frozen_model(spec: CheckpointSpec, device: str = "cpu"):
    from src.training.train import build_model_from_config

    # The checkpoint already contains every model weight. Constructing from
    # random initialisation avoids an unnecessary pretrained-weight download,
    # then the complete saved state is restored below.
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(checkpoint["config"])
    config["model"]["freeze_vision"] = True
    config["model"]["freeze_text"] = True
    config["model"]["pretrained_vision"] = False
    config["model"]["pretrained_text"] = False
    config["model"]["vision_backend"] = infer_vision_backend(checkpoint["model_state"])
    model = build_model_from_config(config)
    state_dict = remap_legacy_state_dict(checkpoint["model_state"], set(model.state_dict()))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
        raise RuntimeError("Vision backbone is not frozen")
    if any(parameter.requires_grad for parameter in model.text_encoder.parameters()):
        raise RuntimeError("Text backbone is not frozen")
    return model, config, int(checkpoint.get("epoch", spec.epoch))
