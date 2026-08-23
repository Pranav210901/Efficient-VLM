from __future__ import annotations

from pathlib import Path

VISION_ENCODERS = ("efficientnet_b0", "convnext_tiny", "convnextv2_tiny", "dinov2_vits14", "swin_tiny")
TEXT_ENCODERS = ("minilm_l6", "all_minilm_l6_v2", "bge_small_en", "e5_small_v2", "distilbert")
TEXT_ALIASES = {"minilm_l6": "all_minilm_l6_v2"}
PROMPT_TEMPLATES = (
    "a photo of a {class_name}",
    "an image of a {class_name}",
    "a picture of a {class_name}",
)
DEFAULT_DATASETS = (
    "coco_retrieval", "cifar100_zeroshot", "pets_zeroshot", "eurosat_zeroshot"
)

# Step 11.5 evaluates only these existing BLF checkpoints. The first two are
# the matched-seed finalists from Step 8; the third is the exploratory pairing
# suggested by the strongest baseline multi-task components and its positive
# single-run retrieval delta.
STEP_11_5_BLF_CONFIG_IDS = (
    "dinov2_vits14__all_minilm_l6_v2__local",
    "convnextv2_tiny__all_minilm_l6_v2__local_global",
    "dinov2_vits14__bge_small_en__local",
)


def canonical_text_encoder(name: str) -> str:
    return TEXT_ALIASES.get(name, name)


def output_root(project_root: str | Path) -> Path:
    return Path(project_root) / "results/phase1_multitask"
