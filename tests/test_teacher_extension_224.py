from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from src.alignment_v3.distillation import distillation_loss
from src.alignment_v3.efficiency_frontier import parameter_counts_student
from src.alignment_v3.model import AlignmentV3Model
from src.alignment_v3.training import (
    _assert_parameter_contract,
    _distillation_auxiliary,
)


class _DummyVision(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.output_dim = 384
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


class _DummyText(nn.Module):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.output_dim = 384
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


class _FakeCache:
    def __init__(self, image: torch.Tensor, text: torch.Tensor) -> None:
        self.image = image
        self.text = text
        self.metadata = {"logit_scale": 10.0}

    def lookup(
        self,
        image_ids: list[str],
        text_ids: list[str],
        captions: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_ids, text_ids, captions
        return self.image.to(device), self.text.to(device)


def test_multi_teacher_parameter_contract_and_inference_exclusion() -> None:
    with (
        patch("src.alignment_v3.model.DINOv3VisionEncoder", _DummyVision),
        patch("src.alignment_v3.model.TextEncoder", _DummyText),
    ):
        model = AlignmentV3Model(
            vision_encoder="dummy",
            text_encoder="dummy",
            use_distillation=True,
            teacher_dims={"mobileclip2": 512, "siglip2": 768},
        )
    summary = model.parameter_summary()
    assert summary["params_training_only"] == 983_040
    assert (
        summary["params_trainable_training"] - summary["params_trainable_inference"]
        == 983_040
    )
    assert not any(
        key.startswith(("teacher_heads.", "teacher_image_head.", "teacher_text_head."))
        for key in model.inference_state_dict()
    )
    _assert_parameter_contract(
        model,
        {
            "provenance": {
                "parameter_contract": {
                    "params_trainable_inference": summary["params_trainable_inference"],
                    "params_trainable_training": summary["params_trainable_training"],
                    "params_training_only": 983_040,
                }
            }
        },
        summary,
    )
    try:
        parameter_counts_student(model)
    except AssertionError as exc:
        assert "training-only teacher heads" in str(exc)
    else:
        raise AssertionError(
            "inference parameter reporting accepted a model with teacher heads"
        )


def test_equal_multi_teacher_loss_is_scalar_average() -> None:
    torch.manual_seed(7)
    batch = 4
    outputs = {
        "logit_scale": torch.tensor(14.0),
        "teacher_spaces": {
            "mobileclip2": {
                "image": torch.randn(batch, 512),
                "text": torch.randn(batch, 512),
            },
            "siglip2": {
                "image": torch.randn(batch, 768),
                "text": torch.randn(batch, 768),
            },
        },
    }
    teachers = {
        "mobileclip2": _FakeCache(torch.randn(batch, 512), torch.randn(batch, 512)),
        "siglip2": _FakeCache(torch.randn(batch, 768), torch.randn(batch, 768)),
    }
    config = {
        "distillation": {
            "temperature": 2.0,
            "image_cosine_weight": 0.25,
            "text_cosine_weight": 0.25,
            "kl_weight": 1.0,
            "teachers": [
                {"name": "mobileclip2", "weight": 0.5},
                {"name": "siglip2", "weight": 0.5},
            ],
        }
    }
    ids = [str(index) for index in range(batch)]
    combined, _ = _distillation_auxiliary(
        config, outputs, teachers, ids, ids, ids, torch.device("cpu")
    )
    independent = []
    for name in ("mobileclip2", "siglip2"):
        spaces = outputs["teacher_spaces"][name]
        cache = teachers[name]
        value, _ = distillation_loss(
            {
                "teacher_space_image": spaces["image"],
                "teacher_space_text": spaces["text"],
                "logit_scale": outputs["logit_scale"],
            },
            cache.image,
            cache.text,
            teacher_scale=10.0,
            temperature=2.0,
            image_cosine_weight=0.25,
            text_cosine_weight=0.25,
            kl_weight=1.0,
        )
        independent.append(value)
    torch.testing.assert_close(combined, 0.5 * independent[0] + 0.5 * independent[1])
