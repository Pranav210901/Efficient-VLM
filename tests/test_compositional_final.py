from __future__ import annotations

import json

import torch

from src.alignment_v3.compositional_final import (
    ROOT,
    _jobs,
    _pipeline,
    acquire_winoground,
)
from src.multitask.compositional_evaluator import sugarcrepe_scores, winoground_scores


def test_compositional_roster_matches_final_evaluation() -> None:
    pipeline = _pipeline("configs/compositional_final/pipeline.yaml")
    jobs = _jobs(pipeline)
    assert len(jobs) == 9
    assert sum(job["kind"] == "student" for job in jobs) == 6
    assert sum(job["kind"] == "reference" for job in jobs) == 3
    assert pipeline["evaluation"]["primary_dataset"] == "sugarcrepe"
    assert pipeline["evaluation"]["primary_metric"] == "overall_accuracy"


def test_compositional_preregistration_has_one_primary() -> None:
    value = json.loads((ROOT / "configs/compositional_final/preregistration.json").read_text())
    assert value["status"] == "FROZEN_BEFORE_EVALUATION"
    assert value["primary"] == {
        "dataset": "sugarcrepe",
        "metric": "overall_accuracy",
        "hypothesis": "The dual-LoRA endpoint will exceed the strict-frozen endpoint on compositional caption discrimination.",
        "minimum_practical_effect_pp": 2.0,
    }


def test_metric_implementations_pin_strict_ties_as_incorrect() -> None:
    sugar = sugarcrepe_scores(torch.tensor([1.0, 0.0]), torch.tensor([1.0, -1.0]), ["a", "b"])
    assert sugar["overall_accuracy"] == 0.5
    wino = winoground_scores(torch.tensor([[1.0, 1.0, 0.0, 2.0], [2.0, 0.0, 0.0, 2.0]]))
    assert wino["group_score"] == 0.5


def test_winoground_requires_explicit_licence_acknowledgement() -> None:
    pipeline = _pipeline("configs/compositional_final/pipeline.yaml")
    try:
        acquire_winoground(pipeline, acknowledge_license=False)
    except RuntimeError as error:
        assert "gated" in str(error).lower()
    else:
        raise AssertionError("Winoground acquisition must require explicit acknowledgement")


def test_launcher_has_no_training_path() -> None:
    launcher = (ROOT / "scripts/submit_compositional_final.sh").read_text()
    assert "compositional_final evaluate" not in launcher  # evaluation is inside the sbatch file
    assert "slurm/compositional_final/evaluate.sbatch" in launcher
    assert "train" not in launcher
