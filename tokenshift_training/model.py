from __future__ import annotations

from typing import Any

from freezeshift.model_adaptation import build_model as build_freezeshift_model
from src.alignment_v3.model import AlignmentV3Model
from tokenshift.token_reduction import install_internal_token_reduction


def build_model(
    config: dict[str, Any], *, pretrained: bool = True
) -> AlignmentV3Model:
    """Build the declared frozen/LoRA parent and install fixed TokenShift.

    TokenShift is state-dict neutral. Its location is nevertheless part of the
    resolved config, so block-6 and block-8 checkpoints cannot cross-resume.
    """
    model = build_freezeshift_model(config, pretrained=pretrained)
    recipe = config.get("recipe", {})
    merge_after = recipe.get("tokenshift_merge_after_block")
    if merge_after is None:
        raise ValueError("tokenshift_merge_after_block must be explicitly set")
    install_internal_token_reduction(
        model.vision_encoder,
        merge_after_block=int(merge_after),
    )
    return model

