from __future__ import annotations

import torch
from torch import nn
from unittest.mock import patch

from src.alignment_v3.model import (
    AlignmentV3Model,
    CLSMeanPatchFusion,
    LearnedQueryPatchPool,
    SpatialTokenAdapter,
)
from src.alignment_v3.token_projection_profile import EXPECTED_ADDITIONS, VARIANTS
from src.alignment_v3.token_projection_profile import _safe_profile_flops


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_candidate_shapes_and_parameter_counts() -> None:
    torch.manual_seed(9)
    patches = torch.randn(2, 196, 384)
    cls = torch.randn(2, 384)
    shape = (14, 14)
    candidates = {
        "cls_mean_concat": CLSMeanPatchFusion(384),
        "learned_query_attention": LearnedQueryPatchPool(384, 128, 4),
        "transformer_128": SpatialTokenAdapter(
            384,
            384,
            adapter_dim=128,
            heads=4,
            ff_dim=256,
            dropout=0.05,
        ),
    }
    for name, module in candidates.items():
        assert module(patches, cls, shape).shape == (2, 384)
        assert _parameter_count(module) == EXPECTED_ADDITIONS[name]


def test_register_tokens_are_not_accepted_as_patch_tokens() -> None:
    module = CLSMeanPatchFusion(384)
    cls = torch.randn(2, 384)
    patches_plus_registers = torch.randn(2, 200, 384)
    try:
        module(patches_plus_registers, cls, (14, 14))
    except ValueError as exc:
        assert "patch-token count" in str(exc)
    else:
        raise AssertionError("aggregator silently included four register tokens")


def test_profile_roster_contains_baseline_and_all_candidates() -> None:
    assert VARIANTS == (
        ("baseline_cls", "cls"),
        ("A_cls_mean_concat", "cls_mean_concat"),
        ("B_learned_query_attention", "learned_query_attention"),
        ("C_transformer_128", "transformer_128"),
    )


def test_flop_instrumentation_failure_does_not_abort_latency_stage(
    monkeypatch,
) -> None:
    def fail(*_: object, **__: object) -> dict[str, float]:
        raise AttributeError("'NoneType' object has no attribute 'next_functions'")

    monkeypatch.setattr(
        "src.alignment_v3.token_projection_profile._profile_flops", fail
    )
    result = _safe_profile_flops(
        torch.nn.Identity(),
        torch.empty(1),
        torch.empty(1),
        torch.device("cpu"),
    )
    assert result["image_flops"] is None
    assert (
        result["flop_measurement_status"]
        == "UNAVAILABLE_PROFILER_INCOMPATIBILITY"
    )


class _DummyEncoder(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.output_dim = 384


def test_teacher_heads_are_excluded_from_aggregator_inference_state() -> None:
    with (
        patch("src.alignment_v3.model.DINOv3VisionEncoder", _DummyEncoder),
        patch("src.alignment_v3.model.TextEncoder", _DummyEncoder),
    ):
        model = AlignmentV3Model(
            vision_encoder="dummy",
            text_encoder="dummy",
            image_token_aggregation="learned_query_attention",
            use_distillation=True,
            teacher_dim=512,
        )
    keys = tuple(model.inference_state_dict())
    assert any(key.startswith("token_aggregator.") for key in keys)
    assert not any(
        key.startswith(("teacher_image_head.", "teacher_text_head.", "teacher_heads."))
        for key in keys
    )


def test_scaled_transformers_and_native_width_contract() -> None:
    patches = torch.randn(2, 196, 384)
    cls = torch.randn(2, 384)
    two_block = SpatialTokenAdapter(
        384,
        384,
        adapter_dim=256,
        heads=8,
        ff_dim=512,
        blocks=2,
    )
    native = SpatialTokenAdapter(
        384,
        384,
        adapter_dim=384,
        heads=6,
        ff_dim=768,
        blocks=1,
        native_width_identity=True,
    )
    assert two_block(patches, cls, (14, 14)).shape == (2, 384)
    assert native(patches, cls, (14, 14)).shape == (2, 384)
    assert isinstance(native.input_projection, nn.Identity)
    assert isinstance(native.output_projection, nn.Identity)
