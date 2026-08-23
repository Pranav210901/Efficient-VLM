from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.alignment_v3.final_zero_shot import ROOT, _jobs, _pipeline
from src.alignment_v3.fingerprint import Fingerprint, hash_config


def test_frozen_roster_is_six_students_plus_three_references() -> None:
    jobs = _jobs(_pipeline("configs/final_zero_shot/pipeline.yaml"))
    assert len(jobs) == 9
    assert sum(job["kind"] == "student" for job in jobs) == 6
    assert sum(job["kind"] == "reference" for job in jobs) == 3
    assert {job["seed"] for job in jobs if job["kind"] == "student"} == {42, 43, 44}


def test_student_exports_are_inference_only_and_fingerprint_valid() -> None:
    pipeline = _pipeline("configs/final_zero_shot/pipeline.yaml")
    for job in _jobs(pipeline):
        if job["kind"] != "student":
            continue
        path = ROOT / job["checkpoint"]
        assert path.is_file(), path
        export = torch.load(path, map_location="cpu", weights_only=False)
        fingerprint = Fingerprint.from_dict(export["fingerprint"])
        training_config = copy.deepcopy(export["config"])
        training_config["recipe"] = copy.deepcopy(export["training_recipe"])
        assert hash_config(training_config) == fingerprint.config_hash
        assert export["training_only_heads_removed"] is True
        assert export["config"]["recipe"]["distillation"] is False
        assert not any(
            key.startswith(("teacher_image_head.", "teacher_text_head.", "teacher_heads."))
            for key in export["model_state"]
        )
        assert int(export["parameter_summary"]["params_trainable_inference"]) == int(
            job["expected_inference_trainable_parameters"]
        )


def test_launcher_contains_no_training_stage() -> None:
    launcher = (ROOT / "scripts/submit_final_zero_shot.sh").read_text()
    assert "slurm/final_zero_shot/evaluate.sbatch" in launcher
    assert "slurm/final_zero_shot/report.sbatch" in launcher
    assert "slurm/final_zero_shot/train" not in launcher
    assert "final_zero_shot train" not in launcher


def test_primary_flickr_test_is_structurally_expected() -> None:
    pipeline = _pipeline("configs/final_zero_shot/pipeline.yaml")
    spec = pipeline["datasets"]["flickr30k_test"]
    assert spec["expected_images"] == 1000
    assert spec["expected_captions"] == 5000
    assert Path(spec["csv"]).as_posix() == "data/flickr30k/test.csv"
