from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TokenLayout:
    input_height: int
    input_width: int
    prefix_tokens: int
    patch_tokens_before: int
    patch_tokens_after: int
    total_tokens_before: int
    total_tokens_after: int


def merge_spatial_tokens_2x2(
    tokens: torch.Tensor,
    *,
    prefix_tokens: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Average each non-overlapping 2x2 patch neighbourhood.

    CLS/register prefix tokens are copied exactly and never pooled. The input
    image remains 224x224; only the internal patch sequence is shortened.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected [B,N,C] tokens, got {tuple(tokens.shape)}")
    if height % 2 or width % 2:
        raise ValueError(f"2x2 merge requires an even patch grid, got {(height, width)}")
    expected = prefix_tokens + height * width
    if tokens.shape[1] != expected:
        raise ValueError(
            f"token count {tokens.shape[1]} does not match {prefix_tokens} prefixes "
            f"plus a {height}x{width} patch grid"
        )
    prefix = tokens[:, :prefix_tokens]
    patches = tokens[:, prefix_tokens:].transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], height, width
    )
    # A fixed spatial average is intentionally used instead of content-adaptive
    # ToMe matching: the graph stays regular and batched latency remains auditable.
    patches = F.avg_pool2d(patches, kernel_size=2, stride=2)
    reduced_height, reduced_width = height // 2, width // 2
    patches = patches.flatten(2).transpose(1, 2)
    return torch.cat((prefix, patches), dim=1), (reduced_height, reduced_width)


class InternalTokenReductionController:
    """Mutable profile/training control for one patched timm DINOv3 backbone."""

    def __init__(self, backbone: nn.Module, merge_after_block: int | None) -> None:
        self.backbone = backbone
        self._merge_after_block: int | None = None
        self.set_merge_after_block(merge_after_block)

    @property
    def merge_after_block(self) -> int | None:
        return self._merge_after_block

    def set_merge_after_block(self, value: int | None) -> None:
        if value is not None:
            value = int(value)
            block_count = len(getattr(self.backbone, "blocks", ()))
            if not 1 <= value < block_count:
                raise ValueError(
                    f"merge_after_block must be in [1,{block_count - 1}], got {value}"
                )
        self._merge_after_block = value


def _reduced_forward_features(
    self: nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    controller: InternalTokenReductionController = self._tokenshift_controller
    merge_after = controller.merge_after_block
    if merge_after is None:
        return self._tokenshift_original_forward_features(
            x, attn_mask=attn_mask, is_causal=is_causal
        )
    if attn_mask is not None or is_causal:
        raise ValueError("TokenShift does not support attention masks or causal vision attention")
    if getattr(self, "rope_mixed", False):
        raise ValueError("TokenShift has not registered a mixed-RoPE reduction rule")
    if not bool(getattr(self, "dynamic_img_size", False)):
        raise ValueError("TokenShift requires DINOv3 dynamic image-size handling")

    x = self.patch_embed(x)
    if x.ndim != 4:
        raise ValueError(f"expected NHWC patch grid from DINOv3, got {tuple(x.shape)}")
    batch, height, width, _ = x.shape
    if (height, width) != (14, 14):
        raise ValueError(
            "TokenShift is frozen to a 224px ViT-S/16 input (14x14 patches); "
            f"observed {(height, width)}"
        )
    x, rope = self._pos_embed(x)
    x = self.norm_pre(x)
    prefix_tokens = int(getattr(self, "num_prefix_tokens", 0))
    if prefix_tokens != 5:
        raise ValueError(
            f"expected DINOv3 CLS + four register tokens (5 prefixes), got {prefix_tokens}"
        )

    current_height, current_width = height, width
    for block_number, block in enumerate(self.blocks, start=1):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(
                block,
                x,
                rope=rope,
                attn_mask=None,
                is_causal=False,
                use_reentrant=False,
            )
        else:
            x = block(x, rope=rope, attn_mask=None, is_causal=False)
        if block_number == merge_after:
            x, (current_height, current_width) = merge_spatial_tokens_2x2(
                x,
                prefix_tokens=prefix_tokens,
                height=current_height,
                width=current_width,
            )
            # Later DINOv3 blocks must use positions for the reduced 7x7 grid.
            # Retaining the original 14x14 RoPE would be a silent semantic defect.
            if self.rope is None:
                raise ValueError("DINOv3 rotary position embedding is unexpectedly absent")
            rope = self.rope.get_embed(shape=(current_height, current_width))
    return self.norm(x)


def install_internal_token_reduction(
    vision_encoder: nn.Module,
    *,
    merge_after_block: int | None,
) -> InternalTokenReductionController:
    """Install one state-dict-neutral reduction point into DINOv3."""
    backbone = getattr(vision_encoder, "model", None)
    if backbone is None:
        raise TypeError("vision encoder does not expose its timm model")
    existing = getattr(backbone, "_tokenshift_controller", None)
    if existing is not None:
        existing.set_merge_after_block(merge_after_block)
        return existing
    if type(backbone).__name__ != "Eva" or len(getattr(backbone, "blocks", ())) != 12:
        raise TypeError(
            "TokenShift is scoped to the 12-block timm EVA implementation used by "
            "vit_small_patch16_dinov3.lvd1689m"
        )
    controller = InternalTokenReductionController(backbone, merge_after_block)
    backbone._tokenshift_original_forward_features = backbone.forward_features
    backbone._tokenshift_controller = controller
    backbone.forward_features = MethodType(_reduced_forward_features, backbone)
    return controller


def expected_layout(merge_after_block: int | None) -> TokenLayout:
    reduced = merge_after_block is not None
    patch_tokens_after = 49 if reduced else 196
    return TokenLayout(
        input_height=224,
        input_width=224,
        prefix_tokens=5,
        patch_tokens_before=196,
        patch_tokens_after=patch_tokens_after,
        total_tokens_before=201,
        total_tokens_after=patch_tokens_after + 5,
    )
