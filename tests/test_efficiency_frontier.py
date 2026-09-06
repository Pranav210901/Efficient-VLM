from __future__ import annotations

import torch
from torch import nn

from src.alignment_v3.efficiency_frontier import (
    latency_pareto,
    load_pipeline,
    missing_v4_jobs,
    parameter_counts_reference,
    parameter_counts_student,
    parameter_pareto,
    profiling_jobs,
    student_jobs,
)


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(7, 5)
        self.text_encoder = nn.Linear(11, 5)
        self.vision_projection = nn.Linear(5, 3)
        self.text_projection = nn.Linear(5, 3)


class _ReferenceCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(7, 3)
        self.text = nn.Linear(11, 3)
        self.logit_scale = nn.Parameter(torch.ones(()))


class _Reference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ReferenceCore()


def test_manifest_has_approved_roster_and_only_missing_v4_seeds() -> None:
    pipeline = load_pipeline("configs/efficiency_frontier/pipeline.yaml")
    assert len(student_jobs(pipeline)) == 9
    assert len(profiling_jobs(pipeline)) == 9
    assert [job["seed"] for job in missing_v4_jobs(pipeline)] == [43, 44]
    assert pipeline["historical_v4_completion"]["protected_seed"] == 42
    assert pipeline["datasets"]["primary"]["id"] == "flickr30k_karpathy_test"


def test_loaded_parameter_counts_include_full_stack_and_query_side() -> None:
    student = _Student()
    student_counts = parameter_counts_student(student)  # type: ignore[arg-type]
    assert student_counts["full_stack_inference_parameters"] == sum(
        value.numel() for value in student.parameters()
    )
    assert student_counts["query_side_inference_parameters"] == sum(
        value.numel()
        for module in (student.text_encoder, student.text_projection)
        for value in module.parameters()
    )

    reference = _Reference()
    reference_counts = parameter_counts_reference(reference)  # type: ignore[arg-type]
    assert reference_counts["full_stack_inference_parameters"] == sum(
        value.numel() for value in reference.parameters()
    )
    assert reference_counts["query_side_inference_parameters"] == sum(
        value.numel() for value in reference.model.text.parameters()
    )


def test_pareto_rules() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "entry_id": "a",
                "full_stack_inference_parameters": 10,
                "score": 0.4,
                "latency_median_ms": 4.0,
                "latency_q1_ms": 3.8,
                "latency_q3_ms": 4.2,
            },
            {
                "entry_id": "b",
                "full_stack_inference_parameters": 20,
                "score": 0.3,
                "latency_median_ms": 5.0,
                "latency_q1_ms": 4.8,
                "latency_q3_ms": 5.2,
            },
        ]
    )
    assert parameter_pareto(frame, "score") == {"a"}
    assert latency_pareto(frame, "score", iqr_aware=False) == {"a"}
    assert latency_pareto(frame, "score", iqr_aware=True) == {"a"}


def test_operator_level_flops_change_with_image_resolution() -> None:
    from src.alignment_v3.efficiency_frontier import _profile_flops

    class TinyPaired(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3)
            self.text = nn.Embedding(32, 4)

        def encode_image(self, value: torch.Tensor) -> torch.Tensor:
            return self.conv(value).mean(dim=(2, 3))

        def encode_text_tokens(self, value: torch.Tensor) -> torch.Tensor:
            return self.text(value).mean(dim=1)

    model = TinyPaired().eval()
    tokens = torch.ones(1, 8, dtype=torch.long)
    small = _profile_flops(
        model, torch.ones(1, 3, 16, 16), tokens, torch.device("cpu")
    )
    large = _profile_flops(
        model, torch.ones(1, 3, 32, 32), tokens, torch.device("cpu")
    )
    assert large["image_flops"] > small["image_flops"]
    assert large["caption_flops"] == small["caption_flops"]
    assert large["flop_measurement_version"] == "operator_level_v2"
