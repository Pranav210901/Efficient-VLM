"""Canonical configuration identifiers shared by Phase 1 and Phase 1.5.

Historical result files use a mixture of display labels, checkpoint directory
names, aliases, booleans, and seed-sweep comparison names.  This module is the
single normalisation boundary; derived analyses should never invent their own
string parsing rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VISION_ALIASES = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "convnexttiny": "convnext_tiny",
    "convnextv2_tiny": "convnextv2_tiny",
    "convnext_v2_tiny": "convnextv2_tiny",
    "convnextv2tiny": "convnextv2_tiny",
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vit_s_14": "dinov2_vits14",
    "dino_v2_vits14": "dinov2_vits14",
    "dinov2_small": "dinov2_vits14",
    "swin_tiny": "swin_tiny",
    "swintiny": "swin_tiny",
}

TEXT_ALIASES = {
    "minilm": "all_minilm_l6_v2",
    "minilm_l6": "all_minilm_l6_v2",
    "all_minilm_l6_v2": "all_minilm_l6_v2",
    "sentence_transformers_all_minilm_l6_v2": "all_minilm_l6_v2",
    "bge": "bge_small_en",
    "bge_small": "bge_small_en",
    "bge_small_en": "bge_small_en",
    "bge_small_en_v1_5": "bge_small_en",
    "e5": "e5_small_v2",
    "e5_small": "e5_small_v2",
    "e5_small_v2": "e5_small_v2",
    "distilbert": "distilbert",
    "distilbert_base_uncased": "distilbert",
}

VARIANTS = ("baseline", "local", "global", "local_global")


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("+", "_plus_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_vision_encoder(value: Any) -> str:
    slug = _slug(value).replace("_plus_", "_")
    if slug in VISION_ALIASES:
        return VISION_ALIASES[slug]
    for alias in sorted(VISION_ALIASES, key=len, reverse=True):
        if slug == alias or slug.startswith(alias + "_"):
            return VISION_ALIASES[alias]
    raise ValueError(f"Unknown vision encoder label: {value!r}")


def canonical_text_encoder(value: Any) -> str:
    slug = _slug(value).replace("_plus_", "_")
    if slug in TEXT_ALIASES:
        return TEXT_ALIASES[slug]
    for alias in sorted(TEXT_ALIASES, key=len, reverse=True):
        if slug == alias or slug.startswith(alias + "_"):
            return TEXT_ALIASES[alias]
    raise ValueError(f"Unknown text encoder label: {value!r}")


def canonical_variant(
    value: Any = None,
    *,
    use_local_blf: Any | None = None,
    use_global_blf: Any | None = None,
) -> str:
    if use_local_blf is not None or use_global_blf is not None:
        local = _as_bool(use_local_blf)
        global_ = _as_bool(use_global_blf)
        if local and global_:
            return "local_global"
        if local:
            return "local"
        if global_:
            return "global"
        return "baseline"
    slug = _slug(value)
    if not slug or slug in {"baseline", "base", "none", "no_blf", "baseline_only"}:
        return "baseline"
    if slug in {"local_global", "local_plus_global", "local_and_global", "both", "blf_both"}:
        return "local_global"
    if "local" in slug and "global" in slug:
        return "local_global"
    if slug in {"local", "local_only", "blf_local"} or slug.endswith("_local"):
        return "local"
    if slug in {"global", "global_only", "blf_global"} or slug.endswith("_global"):
        return "global"
    if slug == "only":
        raise ValueError("Variant label 'only' requires local/global context")
    raise ValueError(f"Unknown BLF variant label: {value!r}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class CanonicalConfiguration:
    vision_encoder: str
    text_encoder: str
    variant: str

    @property
    def config_id(self) -> str:
        return f"{self.vision_encoder}__{self.text_encoder}__{self.variant}"


def build_configuration_id(vision_encoder: Any, text_encoder: Any, variant: Any = "baseline", **flags: Any) -> str:
    return CanonicalConfiguration(
        canonical_vision_encoder(vision_encoder),
        canonical_text_encoder(text_encoder),
        canonical_variant(variant, **flags),
    ).config_id


def parse_configuration_id(value: str) -> CanonicalConfiguration:
    text = str(value).strip()
    if "__" in text:
        parts = text.split("__")
        if len(parts) != 3:
            raise ValueError(f"Invalid canonical configuration ID: {value!r}")
        return CanonicalConfiguration(
            canonical_vision_encoder(parts[0]),
            canonical_text_encoder(parts[1]),
            canonical_variant(parts[2]),
        )
    name = Path(text).name
    for suffix in (".pt", ".csv", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = re.sub(r"^(job_\d+_)", "", name)
    name = re.sub(r"_seed\d+$", "", name)
    variant = next((item for item in ("local_global", "baseline", "global", "local") if name.endswith("_" + item)), None)
    if variant is None:
        raise ValueError(f"Cannot identify a BLF variant in {value!r}")
    stem = name[: -(len(variant) + 1)]
    vision_alias = next(
        (alias for alias in sorted(VISION_ALIASES, key=len, reverse=True) if stem.startswith(alias + "_")),
        None,
    )
    if vision_alias is None:
        raise ValueError(f"Cannot identify a vision encoder in {value!r}")
    text = stem[len(vision_alias) + 1 :]
    return CanonicalConfiguration(
        canonical_vision_encoder(vision_alias),
        canonical_text_encoder(text),
        canonical_variant(variant),
    )


def canonical_configuration_id(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, str):
        try:
            return parse_configuration_id(value).config_id
        except ValueError:
            raise
    existing = value.get("config_id")
    if existing:
        return parse_configuration_id(str(existing)).config_id
    vision = value.get("vision_encoder") or value.get("vision")
    text = value.get("text_encoder") or value.get("text")
    if vision and text:
        variant = value.get("variant")
        if variant is None:
            variant = "baseline"
        return build_configuration_id(
            vision,
            text,
            variant,
            use_local_blf=value.get("use_local_blf"),
            use_global_blf=value.get("use_global_blf"),
        )
    run_name = value.get("run_name") or value.get("run_dir") or value.get("best_checkpoint")
    if run_name:
        return parse_configuration_id(str(run_name)).config_id
    raise ValueError(f"Cannot build a canonical configuration ID from fields: {sorted(value)}")
