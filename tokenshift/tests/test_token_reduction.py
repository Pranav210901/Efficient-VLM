from __future__ import annotations

import torch
import timm

from src.alignment_v3.model import DINOv3VisionEncoder
from tokenshift.token_reduction import (
    expected_layout,
    install_internal_token_reduction,
    merge_spatial_tokens_2x2,
)


def test_declared_layout_distinguishes_control_from_reduced_arms() -> None:
    assert expected_layout(None).patch_tokens_after == 196
    assert expected_layout(None).total_tokens_after == 201
    assert expected_layout(8).patch_tokens_after == 49
    assert expected_layout(8).total_tokens_after == 54


def _encoder() -> DINOv3VisionEncoder:
    return DINOv3VisionEncoder("dinov3_vits16", pretrained=False, freeze=True).eval()


def test_prefix_tokens_are_preserved_exactly() -> None:
    tokens = torch.randn(2, 201, 384)
    reduced, shape = merge_spatial_tokens_2x2(
        tokens, prefix_tokens=5, height=14, width=14
    )
    assert reduced.shape == (2, 54, 384)
    assert shape == (7, 7)
    assert torch.equal(reduced[:, :5], tokens[:, :5])


def test_disabled_controller_preserves_original_output_bitwise() -> None:
    torch.manual_seed(4)
    encoder = _encoder()
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        expected = encoder.model.forward_features(images)
        install_internal_token_reduction(
            encoder, merge_after_block=None
        )
        observed = encoder.model.forward_features(images)
    assert torch.equal(expected, observed)


def test_block_6_and_8_produce_49_patches_and_keep_parameter_count() -> None:
    encoder = _encoder()
    before = sum(parameter.numel() for parameter in encoder.parameters())
    state_names = tuple(encoder.state_dict())
    controller = install_internal_token_reduction(
        encoder, merge_after_block=8
    )
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        for block in (8, 6):
            controller.set_merge_after_block(block)
            features = encoder.forward_features(images)
            assert features.local_tokens.shape == (2, 49, 384)
            assert features.spatial_shape == (7, 7)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == before
    assert tuple(encoder.state_dict()) == state_names


def test_non_224_input_fails_loudly() -> None:
    encoder = _encoder()
    install_internal_token_reduction(encoder, merge_after_block=8)
    try:
        encoder.forward_features(torch.randn(1, 3, 192, 192))
    except ValueError as exc:
        assert "224px" in str(exc)
    else:
        raise AssertionError("TokenShift silently accepted a non-224 input")
