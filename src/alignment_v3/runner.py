from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn.functional as F

from src.alignment_v3.distillation import (
    TeacherCache,
    cache_is_fresh,
    distillation_loss,
    save_teacher_cache,
)
from src.alignment_v3.fingerprint import (
    Fingerprint,
    code_fingerprint,
    hash_config,
    hash_payload,
    is_fresh,
    read_fingerprint,
    runtime_versions,
    sha256_file,
    write_fingerprint,
)
from src.alignment_v3.model import DINO_VISION_MODELS, build_model
from src.alignment_v3.references import (
    REFERENCE_CHECKPOINTS,
    build_reference,
    native_loader,
    preprocessing_description,
)
from src.alignment_v3.splits import create_development_split, read_caption_rows, unique_image_ids
from src.alignment_v3.training import load_training_checkpoint, train
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.training.evaluate import extract_embeddings
from src.training.losses import multi_positive_contrastive_loss
from src.training.metrics import compute_retrieval_metrics
from src.data import build_dataloaders
from src.utils.config import deep_update, load_config
from src.utils.device import get_device


ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "src/alignment_v3/fingerprint.py",
    "src/alignment_v3/splits.py",
    "src/alignment_v3/model.py",
    "src/alignment_v3/references.py",
    "src/alignment_v3/distillation.py",
    "src/alignment_v3/training.py",
    "src/alignment_v3/runner.py",
    "src/data/datasets.py",
    "src/training/losses.py",
    "src/training/metrics.py",
)


def load_pipeline(path: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    if not value.is_file():
        raise FileNotFoundError(value)
    return value, load_config(value)


def output_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline.get("output_root", "results/alignment_v3"))


def checkpoint_root(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline.get("checkpoint_root", "checkpoints/alignment_v3"))


def _index(value: int | None) -> int:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if value is None else int(value)
    if selected < 0:
        raise ValueError("array index must be non-negative")
    return selected


def _select(values: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    selected = _index(index)
    if selected >= len(values):
        raise IndexError(f"index {selected} outside manifest of size {len(values)}")
    return values[selected]


def pair_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = pipeline["pair_matrix"]
    jobs = []
    batch_sizes = matrix.get("batch_sizes", [None])
    for vision in matrix["vision_encoders"]:
        for text in matrix["text_encoders"]:
            pair_id = f"{vision}__{text}"
            for batch_size in batch_sizes:
                experiment_id = pair_id
                for seed in matrix.get("seeds", [42]):
                    run_id = f"{experiment_id}__seed_{int(seed)}"
                    if batch_size is not None:
                        run_id = f"{run_id}__b{int(batch_size)}"
                    job = {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "vision_encoder": str(vision),
                        "text_encoder": str(text),
                        "seed": int(seed),
                    }
                    if batch_size is not None:
                        job["batch_size"] = int(batch_size)
                    jobs.append(job)
    return jobs


def recovery_confirmation_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    recovery = pipeline.get("recovery")
    if not recovery:
        return []
    shortlist_path = output_root(pipeline) / "selection/batch_shortlist.json"
    if not shortlist_path.is_file():
        raise RuntimeError("recovery batch shortlist has not completed")
    shortlist = json.loads(shortlist_path.read_text())
    batches = [int(value) for value in shortlist["batch_sizes"]]
    matrix = pipeline["pair_matrix"]
    vision = str(matrix["vision_encoders"][0])
    text = str(matrix["text_encoders"][0])
    pair_id = f"{vision}__{text}"
    jobs = []
    for batch_size in batches:
        experiment_id = pair_id
        for seed in recovery.get("confirmation_seeds", [43]):
            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "run_id": f"{experiment_id}__seed_{int(seed)}__b{batch_size}",
                    "vision_encoder": vision,
                    "text_encoder": text,
                    "batch_size": batch_size,
                    "seed": int(seed),
                }
            )
    return jobs


def sensitivity_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    if bool(pipeline["distillation_sensitivity"].get("include_matched_baseline", True)):
        for seed in pipeline["distillation_sensitivity"]["seeds"]:
            jobs.append(
                {
                    "experiment_id": "matched_baseline",
                    "run_id": f"matched_baseline__seed_{int(seed)}",
                    "strength": 0.0,
                    "distillation": False,
                    "seed": int(seed),
                }
            )
    for strength in pipeline["distillation_sensitivity"]["strengths"]:
        label = str(strength).replace(".", "p")
        for seed in pipeline["distillation_sensitivity"]["seeds"]:
            jobs.append(
                {
                    "experiment_id": f"distill_strength_{label}",
                    "run_id": f"distill_strength_{label}__seed_{int(seed)}",
                    "strength": float(strength),
                    "distillation": True,
                    "seed": int(seed),
                }
            )
    return jobs


def resolution_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    arm = pipeline.get("resolution_arm", {})
    if not bool(arm.get("enabled", False)):
        return jobs
    for image_size in arm["image_sizes"]:
        for seed in arm["seeds"]:
            jobs.append(
                {
                    "experiment_id": f"resolution_{int(image_size)}",
                    "run_id": f"resolution_{int(image_size)}__seed_{int(seed)}",
                    "image_size": int(image_size),
                    "seed": int(seed),
                }
            )
    return jobs


def _wave0_mode(pipeline: dict[str, Any]) -> bool:
    return bool(pipeline.get("wave0", {}).get("enabled", False))


def _wave1_mode(pipeline: dict[str, Any]) -> bool:
    return bool(pipeline.get("wave1", {}).get("enabled", False))


def _caption_label(value: Any) -> str:
    return "all" if value is None or str(value).lower() == "all" else str(int(value))


def wave0_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for recipe in pipeline.get("wave0", {}).get("screening_recipes", []):
        value = dict(recipe)
        captions = value.get("captions_per_image")
        if str(captions).lower() == "all":
            captions = None
        loss_type = str(value["loss_type"])
        batch_size = int(value["batch_size"])
        caption_label = _caption_label(captions)
        experiment_id = f"{loss_type}__captions_{caption_label}__b{batch_size}"
        jobs.append(
            {
                **value,
                "captions_per_image": captions,
                "memory_queue_size": 16384 if loss_type == "infonce_queue" else 0,
                "sweep_multiplier": 1.0,
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}__seed_42",
                "seed": 42,
            }
        )
    return jobs


def wave0_lr_sweep_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for value in pipeline.get("wave0", {}).get("sigmoid_lr_sweep", []):
        batch_size = int(value["batch_size"])
        multiplier = float(value["sweep_multiplier"])
        label = str(multiplier).replace(".", "p").replace("/", "_")
        experiment_id = f"sigmoid__captions_2__b{batch_size}__lr_{label}"
        jobs.append(
            {
                "loss_type": "sigmoid",
                "captions_per_image": 2,
                "batch_size": batch_size,
                "memory_queue_size": 0,
                "sweep_multiplier": multiplier,
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}__seed_42",
                "seed": 42,
            }
        )
    return jobs


def wave0_lr_control_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    # explicit post-screen LR controls; never used to mutate Wave 0 artifacts
    jobs: list[dict[str, Any]] = []
    for value in pipeline.get("wave0", {}).get("lr_controls", []):
        job = dict(value)
        captions = job.get("captions_per_image")
        if str(captions).lower() == "all":
            captions = None
        multiplier = float(job["sweep_multiplier"])
        multiplier_label = str(multiplier).replace(".", "p")
        loss_type = str(job["loss_type"])
        caption_label = _caption_label(captions)
        batch_size = int(job["batch_size"])
        seed = int(job["seed"])
        experiment_id = (
            f"{loss_type}__captions_{caption_label}__b{batch_size}"
            f"__lr_{multiplier_label}"
        )
        jobs.append(
            {
                **job,
                "captions_per_image": captions,
                "memory_queue_size": 0,
                "sweep_multiplier": multiplier,
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}__seed_{seed}",
                "seed": seed,
            }
        )
    return jobs


def wave0_queue_ablation_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    spec = dict(pipeline.get("wave0", {}).get("queue_capacity_ablation", {}))
    jobs: list[dict[str, Any]] = []
    for capacity in spec.pop("memory_queue_sizes", []):
        queue_size = int(capacity)
        experiment_id = f"infonce_queue__captions_all__b1024__queue_{queue_size}"
        jobs.append(
            {
                **spec,
                "captions_per_image": None,
                "memory_queue_size": queue_size,
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}__seed_42",
            }
        )
    return jobs


def wave0_winner_confirmation_jobs(
    pipeline: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = dict(pipeline.get("wave0", {}).get("winner_confirmation", {}))
    seeds = [int(seed) for seed in spec.pop("seeds", [])]
    captions = spec.get("captions_per_image")
    if str(captions).lower() == "all":
        captions = None
    experiment_id = "infonce_no_queue__captions_all__b1024__lr_3p0"
    return [
        {
            **spec,
            "captions_per_image": captions,
            "memory_queue_size": 0,
            "experiment_id": experiment_id,
            "run_id": f"{experiment_id}__seed_{seed}",
            "seed": seed,
        }
        for seed in seeds
    ]


def wave0_rescreen_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    spec = dict(pipeline.get("wave0", {}).get("rescreen", {}))
    vision_encoders = list(spec.pop("vision_encoders", []))
    text_encoders = list(spec.pop("text_encoders", []))
    jobs: list[dict[str, Any]] = []
    for vision in vision_encoders:
        for text_encoder in text_encoders:
            experiment_id = f"{vision}__{text_encoder}"
            jobs.append(
                {
                    **spec,
                    "vision_encoder": str(vision),
                    "text_encoder": str(text_encoder),
                    "captions_per_image": None,
                    "memory_queue_size": 0,
                    "hardware_profile": "teaching",
                    "precision_mode": "native_bf16",
                    "experiment_id": experiment_id,
                    "run_id": (
                        f"{experiment_id}__seed_42"
                        "__hw_teaching_native_bf16"
                    ),
                }
            )
    return jobs


def wave0_rescreen_confirmation_jobs(
    pipeline: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = dict(pipeline.get("wave0", {}).get("rescreen_confirmation", {}))
    text_encoders = [str(value) for value in spec.pop("text_encoders", [])]
    seeds = [int(value) for value in spec.pop("seeds", [])]
    vision_encoder = str(spec.pop("vision_encoder", ""))
    jobs: list[dict[str, Any]] = []
    for text_encoder in text_encoders:
        experiment_id = f"{vision_encoder}__{text_encoder}"
        for seed in seeds:
            jobs.append(
                {
                    **spec,
                    "vision_encoder": vision_encoder,
                    "text_encoder": text_encoder,
                    "captions_per_image": None,
                    "memory_queue_size": 0,
                    "hardware_profile": "teaching",
                    "precision_mode": "native_bf16",
                    "experiment_id": experiment_id,
                    "run_id": (
                        f"{experiment_id}__seed_{seed}"
                        "__hw_teaching_native_bf16"
                    ),
                    "seed": seed,
                }
            )
    return jobs


def wave0_followup_jobs(
    pipeline: dict[str, Any],
) -> list[tuple[str, int, dict[str, Any]]]:
    groups = (
        ("wave0-lr-control", wave0_lr_control_jobs(pipeline)),
        ("wave0-queue-ablation", wave0_queue_ablation_jobs(pipeline)),
        ("wave0-rescreen", wave0_rescreen_jobs(pipeline)),
    )
    return [
        (stage, index, job)
        for stage, jobs in groups
        for index, job in enumerate(jobs)
    ]


def wave0_lr_calibration_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    # six extra sweep jobs plus the three 1x screening cells run early
    base = [
        {**job, "artifact_stage": "wave0-screen"}
        for job in wave0_jobs(pipeline)
        if job["loss_type"] == "sigmoid"
        and job["captions_per_image"] == 2
    ]
    return base + [
        {**job, "artifact_stage": "wave0-lrsweep"}
        for job in wave0_lr_sweep_jobs(pipeline)
    ]


def wave0_oom_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for loss_type in ("infonce_queue", "infonce_no_queue", "sigmoid"):
        for batch_size in (1024, 2048):
            experiment_id = f"oom__{loss_type}__captions_all__b{batch_size}"
            jobs.append(
                {
                    "loss_type": loss_type,
                    "captions_per_image": None,
                    "batch_size": batch_size,
                    "memory_queue_size": 16384 if loss_type == "infonce_queue" else 0,
                    "sweep_multiplier": 1.0,
                    "experiment_id": experiment_id,
                    "run_id": experiment_id,
                    "seed": 42,
                }
            )
    return jobs


def wave0_selection_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    path = output_root(pipeline) / "selection/finalists.json"
    if not path.is_file():
        raise RuntimeError("Wave 0 finalists have not been selected")
    payload = json.loads(path.read_text())
    if payload.get("status") != "FINALISTS_SELECTED":
        raise RuntimeError(f"Wave 0 finalists unavailable: {payload.get('status')}")
    jobs: list[dict[str, Any]] = []
    for finalist in payload["finalists"]:
        for seed in pipeline["wave0"]["finalist_seeds"]:
            jobs.append(
                {
                    **dict(finalist["recipe"]),
                    "experiment_id": str(finalist["experiment_id"]),
                    "run_id": f"{finalist['experiment_id']}__seed_{int(seed)}",
                    "seed": int(seed),
                }
            )
    return jobs


def ablation_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for recipe in pipeline["ablations"]["recipes"]:
        for seed in pipeline["ablations"]["seeds"]:
            jobs.append(
                {
                    **dict(recipe),
                    "experiment_id": str(recipe["id"]),
                    "run_id": f"{recipe['id']}__seed_{int(seed)}",
                    "seed": int(seed),
                }
            )
    return jobs


def final_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"experiment_id": "final_winner", "run_id": f"final_winner__seed_{int(seed)}", "seed": int(seed)}
        for seed in pipeline["final"]["seeds"]
    ]


def transfer_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        {
            "kind": "reference",
            "experiment_id": reference_id,
            "run_id": f"flickr30k__{reference_id}",
            "reference_id": reference_id,
        }
        for reference_id in pipeline["references"]
    ]
    jobs.extend(
        {
            "kind": "final",
            "experiment_id": "final_winner",
            "run_id": f"flickr30k__final_winner__seed_{job['seed']}",
            "seed": job["seed"],
            "final_run_id": job["run_id"],
        }
        for job in final_jobs(pipeline)
    )
    return jobs


def _manifest_frame(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"array_index": index, **job} for index, job in enumerate(jobs)])


def _lock_v2(output: Path) -> dict[str, Any]:
    paths = [
        ROOT / "results/alignment_v2/report.json",
        ROOT / "results/alignment_v2/results_long.csv",
        ROOT / "results/alignment_v2/manifests/unimodal_jobs.csv",
        *sorted((ROOT / "checkpoints/alignment_v2").glob("*/best.pt")),
    ]
    if len(paths) != 12 or not all(path.is_file() for path in paths):
        raise RuntimeError("canonical Alignment v2 safety-net artifacts are incomplete")
    lock_path = output / "manifests/alignment_v2_immutable_lock.json"
    current = {
        "status": "LOCKED",
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    current["digest"] = hash_payload(current["files"])
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text())
        if previous.get("digest") != current["digest"]:
            raise RuntimeError("Alignment v2 immutable safety-net check failed: an artifact changed")
    else:
        atomic_json(current, lock_path)
    return current


def _make_split(pipeline: dict[str, Any]) -> dict[str, Any]:
    split = pipeline["split"]
    return create_development_split(
        ROOT / split["source_csv"],
        ROOT / split["train_csv"],
        ROOT / split["dev_csv"],
        ROOT / split["manifest"],
        dev_images=int(split["dev_images"]),
        seed=int(split["seed"]),
        final_csv=ROOT / split["final_csv"],
    )


def _validate_reference_registry(pipeline: dict[str, Any]) -> None:
    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip_torch is required for Alignment v3") from exc
    available = set(open_clip.list_pretrained())
    missing = [
        REFERENCE_CHECKPOINTS[value]
        for value in pipeline["references"]
        if value not in REFERENCE_CHECKPOINTS or REFERENCE_CHECKPOINTS[value] not in available
    ]
    if missing:
        raise ValueError(f"unavailable paired checkpoints: {missing}")


def _validate_model_registry(pipeline: dict[str, Any]) -> None:
    import timm

    available = set(timm.list_models(pretrained=True))
    missing = [
        DINO_VISION_MODELS[name]
        for name in pipeline["pair_matrix"]["vision_encoders"]
        if name not in DINO_VISION_MODELS or DINO_VISION_MODELS[name] not in available
    ]
    if missing:
        raise ValueError(f"unavailable timm DINOv3 checkpoints: {missing}")


def validate(pipeline_path: str | Path) -> dict[str, Any]:
    path, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for directory in (output, checkpoint_root(pipeline), ROOT / str(pipeline["log_root"])):
        directory.mkdir(parents=True, exist_ok=True)
    split = _make_split(pipeline)
    lock = _lock_v2(output)
    _validate_reference_registry(pipeline)
    _validate_model_registry(pipeline)
    if _probe_mode(pipeline):
        matrix = pipeline["pair_matrix"]
        if len(matrix["vision_encoders"]) != 1 or len(matrix["text_encoders"]) != 1:
            raise ValueError("probe mode requires one locked vision/text pair")
        teacher_id = configured_teacher_id(pipeline)
        if teacher_id not in REFERENCE_CHECKPOINTS:
            raise ValueError(f"unknown probe teacher {teacher_id!r}")
        teacher_r1 = float(pipeline["probe"]["teacher_r1"])
        if not 0 < teacher_r1 < 1:
            raise ValueError("probe.teacher_r1 must be a fraction in (0, 1)")
        if len(sensitivity_jobs(pipeline)) != 4:
            raise ValueError(
                "faithful v4 probe requires two baseline and two distilled jobs"
            )
    if _wave1_mode(pipeline):
        resolution_distillation_only = (
            pipeline["wave1"].get("design")
            == "selected_resolution_distillation_only"
        )
        expected = (
            {("distill_strength_1p0", seed) for seed in (42, 43, 44)}
            if resolution_distillation_only
            else {
                ("matched_baseline", seed) for seed in (42, 43, 44)
            }
            | {
                ("distill_strength_1p0", seed) for seed in (42, 43, 44)
            }
        )
        observed = {
            (str(job["experiment_id"]), int(job["seed"]))
            for job in sensitivity_jobs(pipeline)
        }
        if observed != expected:
            raise ValueError(
                "Wave 1 job manifest does not match its declared design"
            )
        required = {
            "loss_type": "infonce_no_queue",
            "captions_per_image": None,
            "memory_queue_size": 0,
            "batch_size": 1024,
            "sweep_multiplier": 3.0,
            "image_size": 224 if resolution_distillation_only else 256,
        }
        locked = pipeline["wave1"]["locked_recipe"]
        if any(locked.get(key) != value for key, value in required.items()):
            raise ValueError(f"Wave 1 locked recipe must equal {required}")
    if bool(pipeline.get("resolution_arm", {}).get("enabled", False)):
        jobs = resolution_jobs(pipeline)
        expected = {
            (resolution, seed)
            for resolution in (224, 192)
            for seed in (42, 43, 44)
        }
        observed = {(job["image_size"], job["seed"]) for job in jobs}
        if observed != expected:
            raise ValueError("resolution arm must be 224/192 x seeds 42/43/44")
        if pipeline["pair_matrix"]["vision_encoders"] != ["dinov3_vits16"]:
            raise ValueError("resolution arm is restricted to DINOv3 ViT-S/16")
    if _wave0_mode(pipeline):
        if len(wave0_jobs(pipeline)) != 18:
            raise ValueError("Wave 0 requires exactly 18 screening jobs")
        if len(wave0_lr_sweep_jobs(pipeline)) != 6:
            raise ValueError("Wave 0 requires exactly six additional LR sweep jobs")
        if len(wave0_oom_jobs(pipeline)) != 6:
            raise ValueError("Wave 0 requires exactly six OOM smoke jobs")
        controls = wave0_lr_control_jobs(pipeline)
        expected_controls = {
            ("infonce_no_queue", None, 1024, 3.0, 42),
            ("infonce_no_queue", None, 1024, 6.0, 42),
            ("sigmoid", None, 1024, 6.0, 42),
            ("sigmoid", None, 1024, 10.0, 42),
        }
        observed_controls = {
            (
                str(job["loss_type"]),
                job["captions_per_image"],
                int(job["batch_size"]),
                float(job["sweep_multiplier"]),
                int(job["seed"]),
            )
            for job in controls
        }
        if observed_controls != expected_controls:
            raise ValueError("Wave 0 LR controls must be the approved four-job C/D grid")
        if pipeline["wave0"].get("lr_control_status") not in {
            "PROVISIONAL_PENDING_LR_CONTROL",
            "SELECTED_WAVE0_RECIPE",
        }:
            raise ValueError("Wave 0 has an invalid LR-control status")
        if {
            int(job["memory_queue_size"])
            for job in wave0_queue_ablation_jobs(pipeline)
        } != {1024, 4096, 8192}:
            raise ValueError("Wave 0 queue ablation must contain 1024/4096/8192")
        if len(wave0_rescreen_jobs(pipeline)) != 6:
            raise ValueError("Wave 0 re-screen requires exactly six v3 pairs")
        rescreen_confirmations = wave0_rescreen_confirmation_jobs(pipeline)
        if len(rescreen_confirmations) != 6:
            raise ValueError("Wave 0 top-three confirmation requires six jobs")
        if {
            (job["vision_encoder"], job["text_encoder"], int(job["seed"]))
            for job in rescreen_confirmations
        } != {
            ("dinov3_vits16", text, seed)
            for text in ("e5_small_v2", "bge_small_en", "all_minilm_l6_v2")
            for seed in (43, 44)
        }:
            raise ValueError("top-three confirmation grid does not match approval")
        confirmations = wave0_winner_confirmation_jobs(pipeline)
        if [job["seed"] for job in confirmations] != [43, 44]:
            raise ValueError("Wave 0 winner confirmation requires seeds 43 and 44")
        if any(
            (
                job["loss_type"],
                job["captions_per_image"],
                job["batch_size"],
                job["sweep_multiplier"],
            )
            != ("infonce_no_queue", None, 1024, 3.0)
            for job in [
                *confirmations,
                *wave0_rescreen_jobs(pipeline),
                *rescreen_confirmations,
            ]
        ):
            raise ValueError("confirmation and pair re-screen must use locked recipe")
        if pipeline["pair_matrix"]["vision_encoders"] != ["dinov3_vits16"] or (
            pipeline["pair_matrix"]["text_encoders"] != ["all_minilm_l6_v2"]
        ):
            raise ValueError(
                "locked student must remain DINOv3 ViT-S/16 + MiniLM"
            )
        pair_decision = pipeline["wave0"]["rescreen_confirmation"]
        if (
            pair_decision.get("adoption_status") != "DECIDED_KEEP_MINILM"
            or pair_decision.get("student_pair_changed") is not False
            or pair_decision.get("student_pair_decision_permanent_for_study")
            is not True
        ):
            raise ValueError("main-study pair decision must permanently keep MiniLM")
        prediction_path = ROOT / str(pipeline["wave0"]["predictions_path"])
        predictions = json.loads(prediction_path.read_text())
        ids = {str(value["id"]) for value in predictions.get("predictions", [])}
        required_ids = {"replication_anchor", "wave0_winner", "batch_effect"}
        if ids != required_ids:
            raise ValueError(f"Wave 0 prediction ids must be {sorted(required_ids)}")
    jobs_by_stage = {
        "reference": [{"reference_id": value} for value in pipeline["references"]],
        "pair": pair_jobs(pipeline),
        "sensitivity": sensitivity_jobs(pipeline),
        "ablation": ablation_jobs(pipeline),
        "final": final_jobs(pipeline),
        "transfer": transfer_jobs(pipeline),
    }
    if bool(pipeline.get("resolution_arm", {}).get("enabled", False)):
        jobs_by_stage["resolution"] = resolution_jobs(pipeline)
    if _wave0_mode(pipeline):
        jobs_by_stage.update(
            {
                "wave0": wave0_jobs(pipeline),
                "wave0_lr_sweep": wave0_lr_sweep_jobs(pipeline),
                "wave0_lr_control": wave0_lr_control_jobs(pipeline),
                "wave0_queue_ablation": wave0_queue_ablation_jobs(pipeline),
                "wave0_winner_confirmation": wave0_winner_confirmation_jobs(pipeline),
                "wave0_rescreen": wave0_rescreen_jobs(pipeline),
                "wave0_rescreen_confirmation": (
                    wave0_rescreen_confirmation_jobs(pipeline)
                ),
                "wave0_oom": wave0_oom_jobs(pipeline),
            }
        )
    for stage, jobs in jobs_by_stage.items():
        atomic_csv(_manifest_frame(jobs), manifests / f"{stage}_jobs.csv")
    missing_optional = []
    flickr = ROOT / str(pipeline["optional_transfer"]["flickr30k_csv"])
    if not flickr.is_file():
        missing_optional.append(str(flickr.relative_to(ROOT)))
    report = {
        "status": "READY",
        "pipeline": str(path.relative_to(ROOT)),
        "split": split,
        "v2_lock_digest": lock["digest"],
        "job_counts": {key: len(value) for key, value in jobs_by_stage.items()},
        "optional_missing": missing_optional,
        "runtime_versions": runtime_versions(),
        "code_fingerprint": code_fingerprint(ROOT, CODE_PATHS),
    }
    atomic_json(report, manifests / "validation.json")
    return report


def _split_hash(config: dict[str, Any], *, full_train: bool = False) -> str:
    split_manifest = str(
        config.get("provenance", {}).get(
            "split_manifest", "results/alignment_v3/manifests/split.json"
        )
    )
    split = json.loads((ROOT / split_manifest).read_text())
    return str(split["train_split_hash"] if not full_train else hash_payload({
        "full_train_sha256": sha256_file(ROOT / config["data"]["full_train_csv"])
    }))


def _fingerprint(config: dict[str, Any], *, full_train: bool = False) -> Fingerprint:
    vision = str(config["model"]["vision_encoder"])
    text = str(config["model"]["text_encoder"])
    from src.models.text_encoders import TEXT_MODEL_REGISTRY

    return Fingerprint(
        config_hash=hash_config(config),
        dataset_split_hash=_split_hash(config, full_train=full_train),
        model_checkpoint_id=f"{DINO_VISION_MODELS[vision]}+{TEXT_MODEL_REGISTRY[text]}",
        cache_version=str(config.get("distillation", {}).get("cache_version", "none")),
        preprocessing_hash=hash_config(
            {
                key: config["data"].get(key)
                for key in ("image_size", "interpolation", "image_mean", "image_std")
            }
        ),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
        seed=int(config["seed"]),
    )


def _same_run_identity_except_code(left: Fingerprint, right: Fingerprint) -> bool:
    # allow evaluation code to advance without changing training identity
    return all(
        getattr(left, field) == getattr(right, field)
        for field in (
            "config_hash",
            "dataset_split_hash",
            "model_checkpoint_id",
            "cache_version",
            "preprocessing_hash",
            "seed",
            "schema_version",
        )
    )


def _wave0_predictions(pipeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = ROOT / str(pipeline["wave0"]["predictions_path"])
    payload = json.loads(path.read_text())
    values = {
        str(value["id"]): dict(value) for value in payload.get("predictions", [])
    }
    if not values or any(not str(value.get("statement", "")).strip() for value in values.values()):
        raise ValueError("Wave 0 predictions are missing or contain an empty statement")
    return values


def _wave0_prediction_ids(job: dict[str, Any], stage: str) -> list[str]:
    if stage == "wave0-screen":
        values = ["wave0_winner"]
        if (
            job["loss_type"] == "infonce_queue"
            and job.get("captions_per_image") == 2
            and int(job["batch_size"]) == 2048
        ):
            values.append("replication_anchor")
        return values
    if stage in {
        "wave0-lrsweep",
        "wave0-select",
        "wave0-lr-control",
        "wave0-winner-confirmation",
    }:
        return ["wave0_winner"]
    return []


def append_ledger_entry(
    pipeline: dict[str, Any],
    *,
    job: dict[str, Any],
    stage: str,
    status: str,
    dev_r1: float | None = None,
    test_r1: float | None = None,
    training_metrics: dict[str, Any] | None = None,
    fingerprint_override: Fingerprint | None = None,
    config_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config_override or build_job_config(pipeline, job, stage)
    fingerprint = fingerprint_override or _fingerprint(config)
    predictions = _wave0_predictions(pipeline)
    metrics = training_metrics or {}
    entries: list[dict[str, Any]] = []
    prediction_ids: list[str | None] = _wave0_prediction_ids(job, stage)
    if not prediction_ids and stage in {
        "wave0-queue-ablation",
        "wave0-rescreen",
        "wave0-rescreen-confirmation",
    }:
        prediction_ids = [None]
    for prediction_id in prediction_ids:
        prediction = predictions[prediction_id] if prediction_id is not None else None
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "wave": (
                "wave0_rescreen"
                if stage in {"wave0-rescreen", "wave0-rescreen-confirmation"}
                else "wave0"
            ),
            "run_id": str(job["run_id"]),
            "question": (
                str(job["question"])
                if stage in {
                    "wave0-lr-control",
                    "wave0-winner-confirmation",
                    "wave0-queue-ablation",
                    "wave0-rescreen",
                    "wave0-rescreen-confirmation",
                }
                else prediction_id
            ),
            "prediction": (
                str(prediction["statement"]) if prediction is not None else None
            ),
            "config_fingerprint": fingerprint.digest,
            "seed": int(job["seed"]),
            "git_sha": code_fingerprint(ROOT, CODE_PATHS),
            "git_sha_is_code_fingerprint": True,
            "batch_size": int(config["training"]["batch_size"]),
            "unique_images_per_batch": metrics.get("unique_images_per_batch"),
            "text_rows_per_batch": metrics.get("text_rows_per_batch"),
            "loss_type": str(config["recipe"].get("loss_type", "infonce_queue")),
            "memory_queue_size": int(config["training"]["memory_queue_size"]),
            "seconds_per_epoch": metrics.get("epoch_seconds"),
            "captions_per_image": (
                "all"
                if config["data"].get("train_captions_per_image") is None
                else int(config["data"]["train_captions_per_image"])
            ),
            "base_lr": float(config["training"]["base_lr"]),
            "sqrt_scale_factor": float(config["training"]["sqrt_scale_factor"]),
            "sweep_multiplier": float(config["training"]["sweep_multiplier"]),
            "resolved_lr": float(config["training"]["lr"]),
            "lr_scale_cap_hit": bool(config["training"]["lr_scale_cap_hit"]),
            "mean_loss_magnitude": metrics.get("mean_loss_magnitude"),
            "dev_r1": dev_r1,
            "test_r1": test_r1,
            "status": str(status),
        }
        entries.append(entry)
    if not entries:
        return []
    ledger = ROOT / "logs/run_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing_keys: set[tuple[str, str, str]] = set()
        handle.seek(0)
        for line in handle:
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(line)
                existing_keys.add(
                    (
                        str(value.get("config_fingerprint")),
                        str(value.get("question")),
                        str(value.get("status")),
                    )
                )
        new_entries = [
            entry
            for entry in entries
            if (
                str(entry["config_fingerprint"]),
                str(entry["question"]),
                str(entry["status"]),
            )
            not in existing_keys
        ]
        handle.seek(0, os.SEEK_END)
        for entry in new_entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return new_entries


def ledger_pending(pipeline_path: str | Path, stage: str, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(_training_stage_jobs(pipeline, stage), index)
    if stage == "wave0-lr-calibration":
        stage = str(job["artifact_stage"])
    entries = append_ledger_entry(
        pipeline, job=job, stage=stage, status="PENDING"
    )
    return {"status": "PENDING_LOGGED", "run_id": job["run_id"], "entries": len(entries)}


def ledger_pending_stage(pipeline_path: str | Path, stage: str) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    count = 0
    for job in _training_stage_jobs(pipeline, stage):
        count += len(
            append_ledger_entry(
                pipeline, job=job, stage=stage, status="PENDING"
            )
        )
    return {"status": "PENDING_LOGGED", "stage": stage, "entries": count}


def _common_batch(pipeline: dict[str, Any], fallback: int) -> int:
    if not bool(pipeline.get("batch_probe", {}).get("enabled", True)):
        return fallback
    path = output_root(pipeline) / "batch_probe/common_batch.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("status") == "COMPLETE":
            return int(payload["common_batch_size"])
    return fallback


def _probe_mode(pipeline: dict[str, Any]) -> bool:
    return bool(pipeline.get("probe", {}).get("enabled", False))


def configured_teacher_id(pipeline: dict[str, Any]) -> str:
    config = load_config(ROOT / str(pipeline["base_config"]))
    return str(
        config.get("distillation", {}).get(
            "teacher_id", "mobileclip2_s0_dfndr2b"
        )
    )


def _selected_pair(pipeline: dict[str, Any]) -> dict[str, str]:
    if _probe_mode(pipeline) or _wave0_mode(pipeline):
        matrix = pipeline["pair_matrix"]
        vision = list(matrix["vision_encoders"])
        text = list(matrix["text_encoders"])
        if len(vision) != 1 or len(text) != 1:
            raise ValueError("probe/Wave 0 mode requires exactly one locked vision/text pair")
        return {
            "vision_encoder": str(vision[0]),
            "text_encoder": str(text[0]),
        }
    path = output_root(pipeline) / "selection/pair.json"
    if not path.is_file():
        raise RuntimeError("pair selection has not completed")
    payload = json.loads(path.read_text())
    if not payload.get("proceed"):
        raise RuntimeError("pair viability gate did not pass")
    return {
        "vision_encoder": str(payload["vision_encoder"]),
        "text_encoder": str(payload["text_encoder"]),
    }


def _selected_batch(pipeline: dict[str, Any]) -> int | None:
    path = output_root(pipeline) / "selection/pair.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    value = payload.get("batch_size")
    return None if value is None else int(value)


def _gate_passed(pipeline: dict[str, Any], name: str) -> bool:
    path = output_root(pipeline) / f"selection/{name}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    if name == "recipe":
        return str(payload.get("status", "")).startswith("SELECTED")
    return bool(payload.get("proceed", False))


def _direct_recovery_recipe_study(pipeline: dict[str, Any]) -> bool:
    return bool(pipeline.get("recovery", {}).get("direct_recipe_study", False))


def _selected_strength(pipeline: dict[str, Any]) -> float:
    path = output_root(pipeline) / "selection/distillation.json"
    if not path.is_file():
        return float(pipeline.get("recovery", {}).get("distillation_strength", 1.0))
    return float(json.loads(path.read_text()).get("strength", 1.0))


def _selected_recipe(pipeline: dict[str, Any]) -> dict[str, Any]:
    path = output_root(pipeline) / "selection/recipe.json"
    if not path.is_file():
        raise RuntimeError("recipe selection has not completed")
    payload = json.loads(path.read_text())
    return dict(payload["recipe"])


def build_job_config(
    pipeline: dict[str, Any],
    job: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    config = load_config(ROOT / str(pipeline["base_config"]))
    runtime_workers = os.environ.get("ALIGNMENT_WAVE0_NUM_WORKERS")
    if runtime_workers is not None:
        config["data"]["num_workers"] = int(runtime_workers)
        config["data"]["persistent_workers"] = int(runtime_workers) > 0
    runtime_precision = os.environ.get("ALIGNMENT_WAVE0_PRECISION")
    if runtime_precision is not None:
        config["training"]["precision"] = str(runtime_precision)
    if _wave0_mode(pipeline) and str(config["training"].get("precision")).lower() == "fp16":
        raise ValueError("Wave 0 follow-up training in FP16 is prohibited")
    runtime_partition = os.environ.get("ALIGNMENT_WAVE0_PARTITION")
    if runtime_partition is not None:
        config.setdefault("provenance", {})["slurm_partition_profile"] = str(
            runtime_partition
        )
    if stage in {"pair", "confirmation"}:
        pair = {"vision_encoder": job["vision_encoder"], "text_encoder": job["text_encoder"]}
        recipe = {"adapter": False, "distillation": False, "lora": False}
    else:
        pair = (
            {
                "vision_encoder": str(pipeline["pair_matrix"]["vision_encoders"][0]),
                "text_encoder": str(pipeline["pair_matrix"]["text_encoders"][0]),
            }
            if (
                _wave1_mode(pipeline)
                or bool(pipeline.get("resolution_arm", {}).get("enabled", False))
            )
            else _selected_pair(pipeline)
        )
        if stage in {
            "wave0-lrsweep",
            "wave0-lr-control",
            "wave0-queue-ablation",
            "wave0-winner-confirmation",
            "wave0-screen",
            "wave0-select",
            "wave0-oom",
        }:
            recipe = {
                "adapter": False,
                "distillation": False,
                "lora": False,
                "loss_type": str(job["loss_type"]),
                "captions_per_image": job.get("captions_per_image"),
                "memory_queue_size": int(job.get("memory_queue_size", 0)),
                "batch_size": int(job["batch_size"]),
                "sweep_multiplier": float(job.get("sweep_multiplier", 1.0)),
            }
        elif stage in {"wave0-rescreen", "wave0-rescreen-confirmation"}:
            pair = {
                "vision_encoder": str(job["vision_encoder"]),
                "text_encoder": str(job["text_encoder"]),
            }
            recipe = {
                "adapter": False,
                "distillation": False,
                "lora": False,
                "loss_type": str(job["loss_type"]),
                "captions_per_image": job.get("captions_per_image"),
                "memory_queue_size": int(job.get("memory_queue_size", 0)),
                "batch_size": int(job["batch_size"]),
                "sweep_multiplier": float(job["sweep_multiplier"]),
            }
        elif stage == "sensitivity":
            recipe = {
                "adapter": False,
                "distillation": bool(job.get("distillation", True)),
                "lora": False,
            }
            if _wave1_mode(pipeline):
                locked = dict(pipeline["wave1"]["locked_recipe"])
                locked.pop("image_size", None)
                recipe = deep_update(recipe, locked)
        elif stage == "resolution":
            recipe = {
                "adapter": False,
                "distillation": False,
                "lora": False,
                **dict(pipeline["resolution_arm"]["locked_recipe"]),
            }
        elif stage == "ablation":
            recipe = {
                "adapter": bool(job["adapter"]),
                "distillation": bool(job["distillation"]),
                "lora": bool(job["lora"]),
            }
        elif stage == "final":
            recipe = _selected_recipe(pipeline)
        else:
            raise ValueError(stage)
    config["model"] = deep_update(
        config["model"],
        {
            "vision_encoder": pair["vision_encoder"],
            "text_encoder": pair["text_encoder"],
        },
    )
    # Resolution is an experiment coordinate. Historical pipelines fall back
    # to the registry default, while new studies persist an explicit value in
    # the resolved config.
    default_image_size = 256 if pair["vision_encoder"] == "dinov3_vits16" else 224
    selected_image_size = job.get("image_size")
    if selected_image_size is None:
        selected_image_size = pipeline.get("experiment", {}).get(
            "image_size", default_image_size
        )
    config["data"]["image_size"] = int(selected_image_size)
    if bool(pipeline.get("resolution_arm", {}).get("enabled", False)):
        image_size = int(config["data"]["image_size"])
        if image_size not in {192, 224} or image_size % 16:
            raise ValueError(f"invalid resolution-arm image size {image_size}")
        patch_tokens = (image_size // 16) ** 2
        config["provenance"] = deep_update(
            config.get("provenance", {}),
            {
                "resolution_guard": {
                    "configured_image_size": image_size,
                    "patch_size": 16,
                    "patch_tokens": patch_tokens,
                    "prefix_tokens": 5,
                    "total_transformer_tokens": patch_tokens + 5,
                    "dataloader_must_match": True,
                    "profiler_must_match": True,
                    "flop_measurement_must_match": True,
                    "report_must_match": True,
                }
            },
        )
    config["recipe"] = deep_update(config["recipe"], recipe)
    split_manifest = str(pipeline["split"]["manifest"])
    if split_manifest != "results/alignment_v3/manifests/split.json":
        config["provenance"] = deep_update(
            config.get("provenance", {}),
            {"split_manifest": split_manifest},
        )
    config["seed"] = int(job["seed"])
    config["experiment_id"] = str(job["experiment_id"])
    config["run_id"] = str(job["run_id"])
    config["training"]["batch_size"] = _common_batch(
        pipeline, int(config["training"]["batch_size"])
    )
    selected_batch = job.get("batch_size")
    if selected_batch is None and stage not in {"pair", "confirmation"}:
        selected_batch = _selected_batch(pipeline)
    if selected_batch is not None:
        config["training"]["batch_size"] = int(selected_batch)
    recipe_batch = recipe.get("batch_size")
    if recipe_batch is not None:
        config["training"]["batch_size"] = int(recipe_batch)
    if "captions_per_image" in recipe:
        config["data"]["train_captions_per_image"] = recipe["captions_per_image"]
    if "memory_queue_size" in recipe:
        config["training"]["memory_queue_size"] = int(recipe["memory_queue_size"])
    base_lr = float(config["training"]["lr"])
    reference_batch = int(config["training"].get("lr_reference_batch_size", 128))
    raw_scale = 1.0
    scale = 1.0
    cap = float(config["training"].get("max_lr_scale", 3.0))
    if str(config["training"].get("lr_scale_rule", "none")) == "sqrt":
        raw_scale = math.sqrt(config["training"]["batch_size"] / reference_batch)
        scale = min(raw_scale, cap)
        config["training"]["lr"] = base_lr * scale
        config["training"]["applied_lr_scale"] = scale
    sweep_multiplier = float(recipe.get("sweep_multiplier", 1.0))
    if (
        stage in {"wave0-screen", "wave0-select"}
        and recipe.get("loss_type") == "sigmoid"
        and sweep_multiplier == 1.0
    ):
        selection_path = output_root(pipeline) / "selection/lr_sweep.json"
        if selection_path.is_file():
            winners = json.loads(selection_path.read_text()).get("per_batch", {})
            sweep_multiplier = float(
                winners.get(str(config["training"]["batch_size"]), {}).get(
                    "sweep_multiplier", 1.0
                )
            )
    config["training"]["lr"] = float(config["training"]["lr"]) * sweep_multiplier
    config["training"]["base_lr"] = base_lr
    config["training"]["sqrt_scale_factor"] = scale
    config["training"]["sweep_multiplier"] = sweep_multiplier
    config["training"]["lr_scale_cap_hit"] = raw_scale > cap
    config["recipe"]["sweep_multiplier"] = sweep_multiplier
    config["training"]["save_dir"] = str(
        (checkpoint_root(pipeline) / stage / str(job["run_id"])).relative_to(ROOT)
    )
    if stage in {"sensitivity", "ablation", "final"}:
        config["distillation"]["strength"] = (
            float(job["strength"]) if stage == "sensitivity" else _selected_strength(pipeline)
        )
    if stage == "final":
        config["data"]["train_csv"] = config["data"]["full_train_csv"]
        config["training"]["epochs"] = int(pipeline["final"]["epochs"])
        config["training"]["select_on_dev"] = False
        config["training"]["early_stopping_patience"] = -1
    return config


def prefetch(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    loaded = []
    for reference_id in pipeline["references"]:
        model = build_reference(reference_id)
        loaded.append(reference_id)
        del model
        gc.collect()
    seen = set()
    for job in pair_jobs(pipeline):
        key = (job["vision_encoder"], job["text_encoder"])
        if key in seen:
            continue
        seen.add(key)
        config = build_job_config(pipeline, job, "pair")
        model = build_model(config)
        loaded.append("+".join(key))
        del model
        gc.collect()
    result = {"status": "COMPLETE", "models": loaded}
    atomic_json(result, output_root(pipeline) / "prefetch.json")
    return result


def _profile(model: torch.nn.Module, image_size: int, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    benchmark = config["benchmark"]
    warmup = int(benchmark.get("warmup", 20))
    iterations = int(benchmark.get("iterations", 100))
    images = torch.randn(1, 3, image_size, image_size, device=device)
    captions = ["a photograph of an everyday object"]
    with torch.inference_mode():
        for _ in range(warmup):
            model.encode_image(images)
            model.encode_text(captions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model.encode_image(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        image_ms = (time.perf_counter() - start) * 1000 / iterations
        start = time.perf_counter()
        for _ in range(iterations):
            model.encode_text(captions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        text_ms = (time.perf_counter() - start) * 1000 / iterations
    return {
        "image_query_latency_ms": image_ms,
        "text_query_latency_ms": text_ms,
        "bidirectional_pair_latency_ms": image_ms + text_ms,
        "peak_inference_memory_bytes": float(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def _ranking_payload(embeddings: dict[str, object], top_k: int) -> dict[str, Any]:
    image = torch.as_tensor(embeddings["image_embeds"]).float()
    text = torch.as_tensor(embeddings["text_embeds"]).float()
    image_paths = [str(value) for value in embeddings["image_paths"]]
    text_paths = [str(value) for value in embeddings["text_image_paths"]]
    k_text = min(top_k, text.shape[0])
    k_image = min(top_k, image.shape[0])
    i2t = []
    for start in range(0, image.shape[0], 256):
        i2t.append((image[start : start + 256] @ text.t()).topk(k_text, dim=1).indices.int())
    t2i = []
    for start in range(0, text.shape[0], 512):
        t2i.append((text[start : start + 512] @ image.t()).topk(k_image, dim=1).indices.int())
    return {
        "i2t_indices": torch.cat(i2t),
        "t2i_indices": torch.cat(t2i),
        "image_paths": image_paths,
        "text_image_paths": text_paths,
    }


def _metrics_from_embeddings(embeddings: dict[str, object], k_values: list[int]) -> dict[str, float]:
    metrics = compute_retrieval_metrics(
        torch.as_tensor(embeddings["image_embeds"]),
        torch.as_tensor(embeddings["text_embeds"]),
        k_values=k_values,
        image_ids=list(embeddings["image_paths"]),
        text_image_ids=list(embeddings["text_image_paths"]),
    )
    metrics["mean_R@1"] = 0.5 * (metrics["i2t_R@1"] + metrics["t2i_R@1"])
    return metrics


def evaluate_reference(pipeline_path: str | Path, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    reference_id = str(_select([{"reference_id": x} for x in pipeline["references"]], index)["reference_id"])
    destination = output_root(pipeline) / "references" / reference_id
    existing = destination / "metrics.json"
    device = get_device("auto")
    model = build_reference(reference_id).to(device)
    config = load_config(ROOT / str(pipeline["base_config"]))
    final_rows = read_caption_rows(ROOT / str(config["data"]["final_csv"]))
    reference_fingerprint = Fingerprint(
        config_hash=hash_config({"evaluation": config["evaluation"], "benchmark": config["benchmark"]}),
        dataset_split_hash=hash_payload(unique_image_ids(final_rows)),
        model_checkpoint_id=":".join(REFERENCE_CHECKPOINTS[reference_id]),
        cache_version="reference-eval-v1",
        preprocessing_hash=hash_config(preprocessing_description(model)),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
    )
    if existing.is_file() and is_fresh(destination / "fingerprint.json", reference_fingerprint):
        return {**json.loads(existing.read_text()), "status": "SKIPPED_FRESH"}
    loader = native_loader(
        model,
        ROOT / str(config["data"]["final_csv"]),
        image_root=ROOT / str(config["data"]["image_root"]),
        batch_size=min(256, int(config["training"]["batch_size"])),
        num_workers=int(config["data"]["num_workers"]),
    )
    embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, list(config["evaluation"]["k_values"]))
    efficiency = _profile(model, model.image_size, device, config)
    params = sum(parameter.numel() for parameter in model.parameters())
    row = {
        "status": "COMPLETE",
        "source": "measured_local",
        "kind": "paired_reference",
        "experiment_id": reference_id,
        **metrics,
        **efficiency,
        "params_total_inference": params,
        "params_trainable_inference": 0,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "software": runtime_versions(),
        "preprocessing": preprocessing_description(model),
    }
    atomic_json(row, existing)
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    write_fingerprint(destination / "fingerprint.json", reference_fingerprint)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(_ranking_payload(embeddings, int(config["evaluation"]["rankings_top_k"])), destination / "rankings.pt")
    return row


def batch_probe(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    device = get_device("auto")
    if device.type != "cuda":
        raise RuntimeError("batch profiling requires a CUDA GPU")
    probe = pipeline["batch_probe"]
    rows = []
    recommended = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for job in pair_jobs(pipeline):
        config = build_job_config(pipeline, job, "pair")
        model = build_model(config).to(device)
        if job["vision_encoder"] == "dinov3_convnext_tiny":
            model = model.to(memory_format=torch.channels_last)
        model.train()
        pair_rows = []
        for batch_size in probe["candidates"]:
            torch.cuda.empty_cache()
            try:
                image_size = int(config["data"]["image_size"])
                images = torch.randn(
                    int(batch_size), 3, image_size, image_size, device=device
                )
                if job["vision_encoder"] == "dinov3_convnext_tiny":
                    images = images.contiguous(memory_format=torch.channels_last)
                captions = ["an image of an object"] * (2 * int(batch_size))
                image_ids = [f"image-{i}" for i in range(int(batch_size))]
                text_ids = [f"image-{i // 2}" for i in range(2 * int(batch_size))]
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(int(probe["warmup"])):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        output = model(images, captions)
                        loss = multi_positive_contrastive_loss(output["logits"], image_ids, text_ids)
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for _ in range(int(probe["iterations"])):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        output = model(images, captions)
                        loss = multi_positive_contrastive_loss(output["logits"], image_ids, text_ids)
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                row = {
                    "experiment_id": job["experiment_id"],
                    "batch_size": int(batch_size),
                    "images_per_second": int(batch_size) * int(probe["iterations"]) / elapsed,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "memory_fraction": torch.cuda.max_memory_allocated(device) / total_memory,
                    "stable": True,
                }
            except torch.OutOfMemoryError:
                row = {
                    "experiment_id": job["experiment_id"],
                    "batch_size": int(batch_size),
                    "images_per_second": 0.0,
                    "peak_memory_bytes": total_memory,
                    "memory_fraction": 1.0,
                    "stable": False,
                }
                torch.cuda.empty_cache()
            pair_rows.append(row)
            rows.append(row)
        eligible = [
            row for row in pair_rows
            if row["stable"] and row["memory_fraction"] <= float(probe["memory_fraction_limit"])
        ]
        if not eligible:
            raise RuntimeError(f"no safe batch size for {job['experiment_id']}")
        maximum = max(row["images_per_second"] for row in eligible)
        efficient = [
            row for row in eligible
            if row["images_per_second"] >= (1 - float(probe["throughput_tolerance"])) * maximum
        ]
        recommended.append(max(int(row["batch_size"]) for row in efficient))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    destination = output_root(pipeline) / "batch_probe"
    atomic_csv(pd.DataFrame(rows), destination / "profiles.csv")
    result = {
        "status": "COMPLETE",
        "common_batch_size": min(recommended),
        "per_pair_recommended": recommended,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_bytes": total_memory,
    }
    atomic_json(result, destination / "common_batch.json")
    return result


def wave0_oom_smoke_test(
    pipeline_path: str | Path, index: int | None
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(wave0_oom_jobs(pipeline), index)
    config = build_job_config(pipeline, job, "wave0-oom")
    destination = output_root(pipeline) / "oom_smoke" / f"{job['run_id']}.json"
    device = get_device("auto")
    if device.type != "cuda":
        raise RuntimeError("Wave 0 OOM smoke testing requires CUDA")
    result: dict[str, Any]
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loader, _ = build_dataloaders(config)
        batch = next(iter(loader))
        images = batch["images"].to(device, non_blocking=True)
        captions = [str(value) for value in batch["captions"]]
        image_ids = [str(value) for value in batch["image_paths"]]
        text_ids = [str(value) for value in batch["text_image_paths"]]
        model = build_model(config).to(device).train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images, captions)
            if job["loss_type"] == "sigmoid":
                loss = __import__(
                    "src.training.losses", fromlist=["sigmoid_contrastive_loss"]
                ).sigmoid_contrastive_loss(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    outputs["sigmoid_logit_scale"],
                    outputs["sigmoid_logit_bias"],
                    image_ids,
                    text_ids,
                )
            else:
                memory = __import__(
                    "src.training.losses", fromlist=["CrossBatchMemory"]
                ).CrossBatchMemory(int(job["memory_queue_size"]))
                loss = __import__(
                    "src.training.losses", fromlist=["contrastive_loss_with_memory"]
                ).contrastive_loss_with_memory(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    outputs["logit_scale"],
                    image_ids,
                    text_ids,
                    memory,
                )
        loss.backward()
        torch.cuda.synchronize(device)
        result = {
            "status": "COMPLETE",
            "loss_type": job["loss_type"],
            "batch_size": int(job["batch_size"]),
            "captions_per_image": "all",
            "unique_images": len(image_ids),
            "text_rows": len(text_ids),
            "loss": float(loss.detach()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "gpu": torch.cuda.get_device_name(device),
        }
    except torch.OutOfMemoryError as exc:
        result = {
            "status": "OOM",
            "loss_type": job["loss_type"],
            "batch_size": int(job["batch_size"]),
            "captions_per_image": "all",
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "error": str(exc),
        }
    atomic_json(result, destination)
    if result["status"] != "COMPLETE":
        raise RuntimeError(
            f"Wave 0 OOM smoke failed for {job['loss_type']} batch={job['batch_size']}"
        )
    return result


def summarize_wave0_oom(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows = []
    for job in wave0_oom_jobs(pipeline):
        path = output_root(pipeline) / "oom_smoke" / f"{job['run_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing OOM smoke result {path}")
        rows.append(json.loads(path.read_text()))
    if any(row.get("status") != "COMPLETE" for row in rows):
        raise RuntimeError("one or more Wave 0 OOM smoke jobs failed")
    frame = pd.DataFrame(rows).sort_values(["loss_type", "batch_size"])
    atomic_csv(frame, output_root(pipeline) / "oom_smoke/summary.csv")
    result = {
        "status": "COMPLETE",
        "jobs": rows,
        "maximum_peak_gpu_memory_bytes": int(frame["peak_gpu_memory_bytes"].max()),
    }
    atomic_json(result, output_root(pipeline) / "oom_smoke/summary.json")
    return result


def _training_stage_jobs(pipeline: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "pair":
        return pair_jobs(pipeline)
    if stage == "confirmation":
        return recovery_confirmation_jobs(pipeline)
    if stage == "sensitivity":
        return sensitivity_jobs(pipeline)
    if stage == "resolution":
        return resolution_jobs(pipeline)
    if stage == "ablation":
        return ablation_jobs(pipeline)
    if stage == "final":
        return final_jobs(pipeline)
    if stage == "wave0-lrsweep":
        return wave0_lr_sweep_jobs(pipeline)
    if stage == "wave0-lr-control":
        return wave0_lr_control_jobs(pipeline)
    if stage == "wave0-queue-ablation":
        return wave0_queue_ablation_jobs(pipeline)
    if stage == "wave0-winner-confirmation":
        return wave0_winner_confirmation_jobs(pipeline)
    if stage == "wave0-rescreen":
        return wave0_rescreen_jobs(pipeline)
    if stage == "wave0-rescreen-confirmation":
        return wave0_rescreen_confirmation_jobs(pipeline)
    if stage == "wave0-lr-calibration":
        return wave0_lr_calibration_jobs(pipeline)
    if stage == "wave0-screen":
        return wave0_jobs(pipeline)
    if stage == "wave0-select":
        return wave0_selection_jobs(pipeline)
    if stage == "wave0-oom":
        return wave0_oom_jobs(pipeline)
    raise ValueError(f"unknown training stage {stage!r}")


def run_wave0_lr_calibration(
    pipeline_path: str | Path,
    index: int | None,
    *,
    evaluate: bool,
    resume: bool = True,
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(wave0_lr_calibration_jobs(pipeline), index)
    artifact_stage = str(job["artifact_stage"])
    candidates = (
        wave0_jobs(pipeline)
        if artifact_stage == "wave0-screen"
        else wave0_lr_sweep_jobs(pipeline)
    )
    actual_index = next(
        position
        for position, candidate in enumerate(candidates)
        if candidate["run_id"] == job["run_id"]
    )
    if evaluate:
        return evaluate_training_stage(pipeline_path, artifact_stage, actual_index)
    return run_training_stage(
        pipeline_path, artifact_stage, actual_index, resume=resume
    )


def run_wave0_followup(
    pipeline_path: str | Path,
    index: int | None,
    *,
    evaluate: bool,
    resume: bool = True,
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    stage, local_index, _ = _select(wave0_followup_jobs(pipeline), index)
    if evaluate:
        return evaluate_training_stage(pipeline_path, stage, local_index)
    return run_training_stage(
        pipeline_path, stage, local_index, resume=resume
    )


def accepted_wave0_followup_jobs(
    pipeline: dict[str, Any],
) -> list[tuple[str, int, dict[str, Any]]]:
    # the seven immutable, completed tasks from Slurm array 2223439
    return wave0_followup_jobs(pipeline)[:7]


def evaluate_accepted_wave0_followup(
    pipeline_path: str | Path, index: int | None
) -> dict[str, Any]:
    # evaluate an accepted checkpoint from array 2223439 without rebuilding config
    _, pipeline = load_pipeline(pipeline_path)
    stage, _, job = _select(accepted_wave0_followup_jobs(pipeline), index)
    checkpoint_dir = checkpoint_root(pipeline) / stage / str(job["run_id"])
    checkpoint = checkpoint_dir / "best.pt"
    config_path = checkpoint_dir / "config.yaml"
    fingerprint = read_fingerprint(checkpoint_dir / "fingerprint.json")
    if not checkpoint.is_file() or not config_path.is_file() or fingerprint is None:
        raise RuntimeError(f"incomplete accepted checkpoint bundle: {checkpoint_dir}")
    config = load_config(config_path)
    if hash_config(config) != fingerprint.config_hash:
        raise RuntimeError(f"saved config/fingerprint mismatch: {checkpoint_dir}")
    destination = output_root(pipeline) / stage / str(job["run_id"])
    existing = destination / "metrics.json"
    if existing.is_file() and is_fresh(destination / "fingerprint.json", fingerprint):
        return {**json.loads(existing.read_text()), "status": "SKIPPED_FRESH"}
    device = get_device("auto")
    model = build_model(config).to(device)
    saved = load_training_checkpoint(
        checkpoint, model, device=device, expected_fingerprint=fingerprint
    )
    eval_config = deep_update(
        config,
        {"training": {"batch_size": int(config["evaluation"].get("batch_size", 256))}},
    )
    _, loader = build_dataloaders(eval_config)
    embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(
        embeddings, list(config["evaluation"]["k_values"])
    )
    row = {
        "status": "COMPLETE",
        "source": "measured_local",
        "kind": stage,
        "experiment_id": str(job["experiment_id"]),
        "run_id": str(job["run_id"]),
        "seed": int(job["seed"]),
        **metrics,
        **_profile(model, int(config["data"]["image_size"]), device, config),
        **model.parameter_summary(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "fingerprint_digest": fingerprint.digest,
        "training_code_version": fingerprint.code_version,
        "evaluation_code_version": code_fingerprint(ROOT, CODE_PATHS),
        "resolved_lr": float(config["training"]["lr"]),
        "sqrt_scale_factor": float(config["training"]["sqrt_scale_factor"]),
        "sweep_multiplier": float(config["training"]["sweep_multiplier"]),
        "memory_queue_size": int(config["training"]["memory_queue_size"]),
        "historical_slurm_array_job": "2223439",
        "historical_slurm_array_index": int(index if index is not None else 0),
        "evaluation_only": True,
    }
    atomic_json(row, existing)
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    write_fingerprint(destination / "fingerprint.json", fingerprint)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        _ranking_payload(embeddings, int(config["evaluation"]["rankings_top_k"])),
        destination / "rankings.pt",
    )
    append_ledger_entry(
        pipeline,
        job=job,
        stage=stage,
        status="COMPLETE",
        dev_r1=float(row["mean_R@1"]),
        training_metrics=dict(saved.get("metrics", {})),
        fingerprint_override=fingerprint,
        config_override=config,
    )
    return row


def run_training_stage(
    pipeline_path: str | Path,
    stage: str,
    index: int | None,
    *,
    resume: bool,
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(_training_stage_jobs(pipeline, stage), index)
    if (
        stage == "wave0-screen"
        and job.get("loss_type") == "sigmoid"
        and job.get("captions_per_image") == 2
        and (output_root(pipeline) / "selection/lr_sweep.json").is_file()
    ):
        return {
            "status": "REUSED_LR_CALIBRATION",
            "run_id": job["run_id"],
            "source_run_id": _wave0_lr_source_job(pipeline, job)["run_id"],
        }
    if stage == "sensitivity" and not (
        _probe_mode(pipeline) or _wave1_mode(pipeline) or _gate_passed(pipeline, "pair")
    ):
        return {"status": "SKIPPED_GATE", "gate": "pair", "run_id": job["run_id"]}
    if stage == "ablation":
        gate = "pair" if _direct_recovery_recipe_study(pipeline) else "distillation"
        if not _gate_passed(pipeline, gate):
            return {"status": "SKIPPED_GATE", "gate": gate, "run_id": job["run_id"]}
    if stage == "final" and not _gate_passed(pipeline, "recipe"):
        return {"status": "SKIPPED_GATE", "gate": "recipe", "run_id": job["run_id"]}
    config = build_job_config(pipeline, job, stage)
    requested_fingerprint = _fingerprint(config, full_train=stage == "final")
    save_dir = ROOT / str(config["training"]["save_dir"])
    existing_fingerprint = read_fingerprint(save_dir / "fingerprint.json")
    fingerprint = requested_fingerprint
    if existing_fingerprint is not None:
        if not _same_run_identity_except_code(
            existing_fingerprint, requested_fingerprint
        ):
            raise RuntimeError(
                "checkpoint directory is occupied by a different runtime/config "
                f"identity; refusing to overwrite or cross-resume: {save_dir}"
            )
        fingerprint = existing_fingerprint
    if (
        (save_dir / "inference.pt").is_file()
        and is_fresh(save_dir / "fingerprint.json", fingerprint)
    ):
        return {"status": "SKIPPED_FRESH", "run_id": job["run_id"]}
    result = train(config, fingerprint, resume=resume)
    return {"run_id": job["run_id"], **result}


def evaluate_training_stage(
    pipeline_path: str | Path,
    stage: str,
    index: int | None,
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(_training_stage_jobs(pipeline, stage), index)
    if (
        stage == "wave0-screen"
        and job.get("loss_type") == "sigmoid"
        and job.get("captions_per_image") == 2
        and (output_root(pipeline) / "selection/lr_sweep.json").is_file()
    ):
        return _reuse_wave0_lr_evaluation(pipeline, job)
    if stage == "sensitivity" and not (
        _probe_mode(pipeline) or _wave1_mode(pipeline) or _gate_passed(pipeline, "pair")
    ):
        return {"status": "SKIPPED_GATE", "gate": "pair", "run_id": job["run_id"]}
    if stage == "ablation":
        gate = "pair" if _direct_recovery_recipe_study(pipeline) else "distillation"
        if not _gate_passed(pipeline, gate):
            return {"status": "SKIPPED_GATE", "gate": gate, "run_id": job["run_id"]}
    if stage == "final" and not _gate_passed(pipeline, "recipe"):
        return {"status": "SKIPPED_GATE", "gate": "recipe", "run_id": job["run_id"]}
    config = build_job_config(pipeline, job, stage)
    destination = output_root(pipeline) / stage / str(job["run_id"])
    existing = destination / "metrics.json"
    evaluation_fingerprint = _fingerprint(config, full_train=stage == "final")
    checkpoint = ROOT / str(config["training"]["save_dir"]) / "best.pt"
    training_fingerprint = read_fingerprint(checkpoint.parent / "fingerprint.json")
    if training_fingerprint is None:
        raise RuntimeError(f"missing training fingerprint for {checkpoint}")
    if not _same_run_identity_except_code(
        training_fingerprint, evaluation_fingerprint
    ):
        raise RuntimeError(
            "evaluation config does not match the trained checkpoint identity: "
            f"{checkpoint}"
        )
    if existing.is_file() and is_fresh(
        destination / "fingerprint.json", training_fingerprint
    ):
        return {**json.loads(existing.read_text()), "status": "SKIPPED_FRESH"}
    device = get_device("auto")
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint,
        model,
        device=device,
        expected_fingerprint=training_fingerprint,
    )
    eval_config = deep_update(
        config,
        {
            "data": {
                "val_csv": config["data"]["final_csv"]
                if stage == "final"
                else config["data"]["val_csv"]
            },
            "training": {"batch_size": int(config["evaluation"].get("batch_size", 256))},
        },
    )
    _, loader = __import__("src.data", fromlist=["build_dataloaders"]).build_dataloaders(eval_config)
    embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, list(config["evaluation"]["k_values"]))
    efficiency = _profile(model, int(config["data"]["image_size"]), device, config)
    parameters = model.parameter_summary()
    row = {
        "status": "COMPLETE",
        "source": "measured_local",
        "kind": stage,
        "experiment_id": str(job["experiment_id"]),
        "run_id": str(job["run_id"]),
        "seed": int(job["seed"]),
        **metrics,
        **efficiency,
        **parameters,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "fingerprint_digest": training_fingerprint.digest,
        "training_code_version": training_fingerprint.code_version,
        "evaluation_code_version": evaluation_fingerprint.code_version,
        "resolved_lr": float(config["training"]["lr"]),
        "sqrt_scale_factor": float(config["training"]["sqrt_scale_factor"]),
        "sweep_multiplier": float(config["training"]["sweep_multiplier"]),
        "memory_queue_size": int(config["training"]["memory_queue_size"]),
    }
    atomic_json(row, existing)
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    write_fingerprint(destination / "fingerprint.json", training_fingerprint)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(_ranking_payload(embeddings, int(config["evaluation"]["rankings_top_k"])), destination / "rankings.pt")
    if _wave0_mode(pipeline) and stage.startswith("wave0-"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        append_ledger_entry(
            pipeline,
            job=job,
            stage=stage,
            status="COMPLETE",
            dev_r1=float(row["mean_R@1"]),
            test_r1=None,
            training_metrics=dict(saved.get("metrics", {})),
            fingerprint_override=training_fingerprint,
        )
    return row


def select_pair(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows = []
    jobs = pair_jobs(pipeline)
    jobs_by_run = {str(job["run_id"]): job for job in jobs}
    for job in jobs:
        path = output_root(pipeline) / "pair" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing pair evaluation {path}")
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    selection = pipeline["selection"]
    best_accuracy = float(frame["mean_R@1"].max())
    equivalent = frame[
        frame["mean_R@1"] >= best_accuracy - float(selection["practical_equivalence_margin"])
    ]
    winner = equivalent.sort_values(
        ["bidirectional_pair_latency_ms", "params_total_inference"], ascending=True
    ).iloc[0]
    gain = float(winner["mean_R@1"]) - float(selection["v2_best_mean_r1"])
    proceed = (
        float(winner["mean_R@1"]) >= float(selection["pair_minimum_mean_r1"])
        or gain >= float(selection["pair_minimum_gain_over_v2"])
    )
    vision, text = str(winner["experiment_id"]).split("__", 1)
    winning_job = jobs_by_run[str(winner["run_id"])]
    result = {
        "status": "SELECTED" if proceed else "STOPPED_AT_GATE",
        "proceed": proceed,
        "experiment_id": winner["experiment_id"],
        "vision_encoder": vision,
        "text_encoder": text,
        "mean_R@1": float(winner["mean_R@1"]),
        "gain_over_v2": gain,
        "equivalence_margin": float(selection["practical_equivalence_margin"]),
    }
    if winning_job.get("batch_size") is not None:
        result["batch_size"] = int(winning_job["batch_size"])
    atomic_json(result, output_root(pipeline) / "selection/pair.json")
    return result


def select_recovery_batches(pipeline_path: str | Path) -> dict[str, Any]:
    # shortlist batch sizes using only the declared seed-42 recovery sweep
    _, pipeline = load_pipeline(pipeline_path)
    recovery = pipeline.get("recovery")
    if not recovery:
        raise RuntimeError("recovery configuration is missing")
    rows = []
    for job in pair_jobs(pipeline):
        path = output_root(pipeline) / "pair" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing recovery sweep evaluation {path}")
        row = json.loads(path.read_text())
        row["batch_size"] = int(job["batch_size"])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["mean_R@1", "batch_size"], ascending=[False, True]
    )
    top_k = min(int(recovery.get("confirmation_top_k", 2)), len(frame))
    selected = frame.head(top_k)
    result = {
        "status": "SELECTED_FOR_CONFIRMATION",
        "batch_sizes": [int(value) for value in selected["batch_size"].tolist()],
        "seed": int(pipeline["pair_matrix"].get("seeds", [42])[0]),
        "confirmation_seeds": [
            int(value) for value in recovery.get("confirmation_seeds", [43])
        ],
        "ranking": [
            {
                "batch_size": int(row["batch_size"]),
                "mean_R@1": float(row["mean_R@1"]),
            }
            for row in frame.to_dict("records")
        ],
    }
    destination = output_root(pipeline) / "selection/batch_shortlist.json"
    atomic_json(result, destination)
    atomic_csv(
        _manifest_frame(recovery_confirmation_jobs(pipeline)),
        output_root(pipeline) / "manifests/confirmation_jobs.csv",
    )
    return result


def select_recovery_pair(pipeline_path: str | Path) -> dict[str, Any]:
    # apply the original pair gate to two-seed fixed-batch recovery results
    _, pipeline = load_pipeline(pipeline_path)
    recovery = pipeline.get("recovery")
    if not recovery:
        raise RuntimeError("recovery configuration is missing")
    shortlist_path = output_root(pipeline) / "selection/batch_shortlist.json"
    if not shortlist_path.is_file():
        raise RuntimeError("recovery batch shortlist has not completed")
    shortlisted = {
        int(value) for value in json.loads(shortlist_path.read_text())["batch_sizes"]
    }
    jobs = [
        job for job in pair_jobs(pipeline) if int(job["batch_size"]) in shortlisted
    ] + recovery_confirmation_jobs(pipeline)
    rows = []
    for job in jobs:
        folder = "pair" if int(job["seed"]) in pipeline["pair_matrix"]["seeds"] else "confirmation"
        path = output_root(pipeline) / folder / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing recovery confirmation evaluation {path}")
        row = json.loads(path.read_text())
        row["batch_size"] = int(job["batch_size"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected_seeds = {
        *[int(value) for value in pipeline["pair_matrix"]["seeds"]],
        *[int(value) for value in recovery.get("confirmation_seeds", [43])],
    }
    counts = frame.groupby("batch_size")["seed"].nunique()
    incomplete = [
        int(batch) for batch, count in counts.items() if int(count) != len(expected_seeds)
    ]
    if incomplete:
        raise RuntimeError(f"incomplete recovery seeds for batch sizes {incomplete}")
    summary = (
        frame.groupby("batch_size", as_index=False)
        .agg(
            mean_R1=("mean_R@1", "mean"),
            min_R1=("mean_R@1", "min"),
            max_R1=("mean_R@1", "max"),
            latency=("bidirectional_pair_latency_ms", "mean"),
            total_params=("params_total_inference", "mean"),
            trainable_params=("params_trainable_inference", "mean"),
        )
        .sort_values(["mean_R1", "batch_size"], ascending=[False, True])
    )
    winner = summary.iloc[0]
    selection = pipeline["selection"]
    mean_r1 = float(winner["mean_R1"])
    gain = mean_r1 - float(selection["v2_best_mean_r1"])
    proceed = (
        mean_r1 >= float(selection["pair_minimum_mean_r1"])
        or gain >= float(selection["pair_minimum_gain_over_v2"])
    )
    matrix = pipeline["pair_matrix"]
    result = {
        "status": "SELECTED" if proceed else "STOPPED_AT_GATE",
        "proceed": proceed,
        "experiment_id": (
            f"{matrix['vision_encoders'][0]}__{matrix['text_encoders'][0]}"
        ),
        "vision_encoder": str(matrix["vision_encoders"][0]),
        "text_encoder": str(matrix["text_encoders"][0]),
        "batch_size": int(winner["batch_size"]),
        "seeds": sorted(expected_seeds),
        "mean_R@1": mean_r1,
        "min_seed_R@1": float(winner["min_R1"]),
        "max_seed_R@1": float(winner["max_R1"]),
        "gain_over_v2": gain,
        "pair_minimum_mean_r1": float(selection["pair_minimum_mean_r1"]),
        "pair_minimum_gain_over_v2": float(selection["pair_minimum_gain_over_v2"]),
    }
    atomic_json(result, output_root(pipeline) / "selection/pair.json")
    atomic_csv(summary, output_root(pipeline) / "selection/batch_confirmation_summary.csv")
    return result


def build_teacher_cache(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    if not (_probe_mode(pipeline) or _wave1_mode(pipeline) or _gate_passed(pipeline, "pair")):
        return {"status": "SKIPPED_GATE", "gate": "pair"}
    config = load_config(ROOT / str(pipeline["base_config"]))
    cache_path = ROOT / str(config["distillation"]["cache_path"])
    cache_csv = str(
        pipeline.get("probe", {}).get(
            "cache_csv",
            config["data"]["train_csv"]
            if (_probe_mode(pipeline) or _wave1_mode(pipeline))
            else config["data"]["full_train_csv"],
        )
    )
    rows = read_caption_rows(ROOT / cache_csv)
    split_hash = hash_payload(
        {
            "image_ids": unique_image_ids(rows),
            "caption_csv_sha256": sha256_file(ROOT / cache_csv),
        }
    )
    teacher_id = str(
        config["distillation"].get("teacher_id", "mobileclip2_s0_dfndr2b")
    )
    if teacher_id not in REFERENCE_CHECKPOINTS:
        raise ValueError(f"unknown distillation teacher {teacher_id!r}")
    teacher_model_name, teacher_pretrained = REFERENCE_CHECKPOINTS[teacher_id]
    fingerprint = Fingerprint(
        config_hash=hash_config(config["distillation"]),
        dataset_split_hash=split_hash,
        model_checkpoint_id=f"{teacher_model_name}:{teacher_pretrained}",
        cache_version=str(config["distillation"]["cache_version"]),
        preprocessing_hash=hash_payload({"native": teacher_id}),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
    )
    if cache_is_fresh(cache_path, fingerprint):
        return {"status": "SKIPPED_FRESH", "path": str(cache_path)}
    device = get_device("auto")
    model = build_reference(teacher_id).to(device)
    expected_dim = int(config["recipe"].get("teacher_dim", model.embedding_dim))
    if model.embedding_dim != expected_dim:
        raise ValueError(
            f"teacher embedding dimension mismatch: {teacher_id} emits "
            f"{model.embedding_dim}, config expects {expected_dim}"
        )
    loader = native_loader(
        model,
        ROOT / cache_csv,
        image_root=ROOT / str(config["data"]["image_root"]),
        batch_size=256,
        num_workers=int(config["data"]["num_workers"]),
    )
    embeddings = extract_embeddings(model, loader, device)
    return save_teacher_cache(
        cache_path,
        image_embeddings=torch.as_tensor(embeddings["image_embeds"]),
        text_embeddings=torch.as_tensor(embeddings["text_embeds"]),
        image_paths=list(embeddings["image_paths"]),
        text_image_paths=list(embeddings["text_image_paths"]),
        captions=list(embeddings["captions"]),
        fingerprint=fingerprint,
        teacher_id=teacher_id,
        logit_scale=model.logit_scale_value,
    )


def select_distillation(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    if not (_probe_mode(pipeline) or _gate_passed(pipeline, "pair")):
        result = {"status": "SKIPPED_GATE", "proceed": False, "gate": "pair"}
        atomic_json(result, output_root(pipeline) / "selection/distillation.json")
        return result
    rows = []
    for job in sensitivity_jobs(pipeline):
        path = output_root(pipeline) / "sensitivity" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing sensitivity result {path}")
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("experiment_id", as_index=False)["mean_R@1"].mean()
    baseline_rows = summary[summary["experiment_id"] == "matched_baseline"]
    if baseline_rows.empty:
        raise RuntimeError("matched two-seed baseline is missing from distillation sensitivity")
    baseline_mean = float(baseline_rows.iloc[0]["mean_R@1"])
    candidates = summary[summary["experiment_id"] != "matched_baseline"]
    winner = candidates.sort_values("mean_R@1", ascending=False).iloc[0]
    strength_label = str(winner["experiment_id"]).removeprefix("distill_strength_").replace("p", ".")
    gain = float(winner["mean_R@1"]) - baseline_mean
    if _probe_mode(pipeline):
        probe = pipeline["probe"]
        teacher_r1 = float(probe["teacher_r1"])
        denominator = teacher_r1 - baseline_mean
        if denominator <= 0:
            raise ValueError(
                "capture fraction requires teacher_R@1 greater than baseline_R@1"
            )
        distilled_mean = float(winner["mean_R@1"])
        baseline_values = frame.loc[
            frame["experiment_id"] == "matched_baseline", "mean_R@1"
        ].astype(float)
        distilled_values = frame.loc[
            frame["experiment_id"] == winner["experiment_id"], "mean_R@1"
        ].astype(float)
        high = float(probe.get("viable_distilled_r1", 0.30))
        low = float(probe.get("nonviable_distilled_r1", 0.22))
        decision = (
            "VIABLE_ABSOLUTE_PATH"
            if distilled_mean >= high
            else "PIVOT_EFFICIENCY_NORMALIZED"
            if distilled_mean <= low
            else "INCONCLUSIVE_MIDDLE"
        )
        result = {
            "status": "COMPLETE",
            "proceed": distilled_mean >= high,
            "teacher_id": str(configured_teacher_id(pipeline)),
            "teacher_R@1": teacher_r1,
            "teacher_anchor_split": str(
                probe.get("teacher_anchor_split", "coco_val_5000")
            ),
            "probe_split": str(probe.get("probe_split", "coco_dev_5000")),
            "baseline_R@1": baseline_mean,
            "baseline_seed_values": [float(value) for value in baseline_values],
            "baseline_seed_std": float(baseline_values.std(ddof=1)),
            "distilled_R@1": distilled_mean,
            "distilled_seed_values": [float(value) for value in distilled_values],
            "distilled_seed_std": float(distilled_values.std(ddof=1)),
            "delta_R@1": gain,
            "capture_fraction": gain / denominator,
            "distillation_strength": float(strength_label),
            "viable_threshold_R@1": high,
            "nonviable_threshold_R@1": low,
            "decision": decision,
            "caveat": (
                "Teacher anchor is measured on sealed COCO-val while the probe "
                "is evaluated on COCO-dev-5000."
            ),
        }
        atomic_json(result, output_root(pipeline) / "selection/distillation.json")
        atomic_json(result, output_root(pipeline) / "probe_report.json")
        return result
    selection = pipeline["selection"]
    proceed = gain >= float(selection["distillation_minimum_gain"]) and float(
        winner["mean_R@1"]
    ) >= float(selection["distillation_minimum_mean_r1"])
    result = {
        "status": "SELECTED" if proceed else "STOPPED_AT_GATE",
        "proceed": proceed,
        "strength": float(strength_label),
        "mean_R@1": float(winner["mean_R@1"]),
        "gain_over_pair": gain,
        "matched_baseline_mean_R@1": baseline_mean,
    }
    atomic_json(result, output_root(pipeline) / "selection/distillation.json")
    return result


def report_wave1(pipeline_path: str | Path) -> dict[str, Any]:
    # descriptive matched-baseline Wave 1 report; no post-hoc gate
    _, pipeline = load_pipeline(pipeline_path)
    if not _wave1_mode(pipeline):
        raise ValueError("report-wave1 requires wave1.enabled")
    rows = []
    for job in sensitivity_jobs(pipeline):
        path = output_root(pipeline) / "sensitivity" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing Wave 1 result {path}")
        row = json.loads(path.read_text())
        rows.append(
            {
                "experiment_id": str(job["experiment_id"]),
                "seed": int(job["seed"]),
                "mean_R@1": float(row["mean_R@1"]),
                "i2t_R@1": float(row["i2t_R@1"]),
                "t2i_R@1": float(row["t2i_R@1"]),
                "run_id": str(job["run_id"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["experiment_id", "seed"])
    summary = (
        frame.groupby("experiment_id", as_index=False)
        .agg(
            mean_R1=("mean_R@1", "mean"),
            sd_R1=("mean_R@1", "std"),
            min_R1=("mean_R@1", "min"),
            max_R1=("mean_R@1", "max"),
        )
        .sort_values("experiment_id")
    )
    baseline = float(
        summary.loc[summary["experiment_id"] == "matched_baseline", "mean_R1"].iloc[0]
    )
    distilled = float(
        summary.loc[
            summary["experiment_id"] == "distill_strength_1p0", "mean_R1"
        ].iloc[0]
    )
    destination = output_root(pipeline) / "wave1_report"
    atomic_csv(frame, destination / "per_seed.csv")
    atomic_csv(summary, destination / "summary.csv")
    result = {
        "status": "COMPLETE",
        "question": str(pipeline["wave1"]["question"]),
        "teacher_id": configured_teacher_id(pipeline),
        "matched_baseline_mean_R1": baseline,
        "distilled_mean_R1": distilled,
        "distillation_delta_pp": 100.0 * (distilled - baseline),
        "seeds": [42, 43, 44],
        "decision": "DESCRIPTIVE_RESULT_NO_POSTHOC_GATE",
        "test_split_used": False,
    }
    atomic_json(result, destination / "report.json")
    return result


def report_wave1_teacher_comparison() -> dict[str, Any]:
    # compare three teacher arms without inventing a post-hoc gate
    specifications = {
        "openclip": {
            "teacher_id": "openclip_vit_b32_quickgelu_openai",
            "reference": ROOT
            / "results/alignment_v3/references/openclip_vit_b32_quickgelu_openai/metrics.json",
        },
        "mobileclip2": {
            "teacher_id": "mobileclip2_s0_dfndr2b",
            "reference": ROOT
            / "results/alignment_v3/references/mobileclip2_s0_dfndr2b/metrics.json",
        },
        "siglip2": {
            "teacher_id": "siglip2_vit_b32_256_webli",
            "reference": ROOT
            / "results/alignment_v3/references/siglip2_vit_b32_256_webli/metrics.json",
        },
    }
    rows: list[dict[str, Any]] = []
    baseline_signatures: list[dict[str, Any]] = []
    for teacher, specification in specifications.items():
        folder = ROOT / "results/alignment_wave1" / teacher
        report_path = folder / "wave1_report/report.json"
        per_seed_path = folder / "wave1_report/per_seed.csv"
        reference_path = Path(specification["reference"])
        if not report_path.is_file() or not per_seed_path.is_file():
            raise RuntimeError(f"missing completed Wave 1 report for {teacher}")
        if not reference_path.is_file():
            raise RuntimeError(f"missing local reference metric {reference_path}")
        report_payload = json.loads(report_path.read_text())
        reference_payload = json.loads(reference_path.read_text())
        frame = pd.read_csv(per_seed_path)
        baseline = frame[frame["experiment_id"] == "matched_baseline"].set_index("seed")
        distilled = frame[
            frame["experiment_id"] == "distill_strength_1p0"
        ].set_index("seed")
        for seed in (42, 43, 44):
            config_path = (
                ROOT
                / "checkpoints/alignment_wave1"
                / teacher
                / "sensitivity"
                / f"matched_baseline__seed_{seed}"
                / "config.yaml"
            )
            config = load_config(config_path)
            baseline_signatures.append(
                {
                    "teacher_arm": teacher,
                    "seed": seed,
                    "vision_encoder": config["model"]["vision_encoder"],
                    "text_encoder": config["model"]["text_encoder"],
                    "train_csv": config["data"]["train_csv"],
                    "val_csv": config["data"]["val_csv"],
                    "image_size": int(config["data"]["image_size"]),
                    "captions_per_image": config["data"]["train_captions_per_image"],
                    "loss_type": config["recipe"]["loss_type"],
                    "batch_size": int(config["training"]["batch_size"]),
                    "memory_queue_size": int(config["training"]["memory_queue_size"]),
                    "resolved_lr": float(config["training"]["lr"]),
                    "precision": config["training"]["precision"],
                }
            )
            rows.append(
                {
                    "teacher_arm": teacher,
                    "teacher_id": specification["teacher_id"],
                    "teacher_standalone_mean_R1": float(
                        reference_payload["mean_R@1"]
                    ),
                    "teacher_strength_coordinate": (
                        "existing locally measured COCO reference anchor"
                    ),
                    "seed": seed,
                    "baseline_mean_R1": float(baseline.loc[seed, "mean_R@1"]),
                    "distilled_mean_R1": float(distilled.loc[seed, "mean_R@1"]),
                    "distillation_gain_pp": 100.0
                    * (
                        float(distilled.loc[seed, "mean_R@1"])
                        - float(baseline.loc[seed, "mean_R@1"])
                    ),
                }
            )
        if report_payload.get("test_split_used") is not False:
            raise RuntimeError(f"Wave 1 arm unexpectedly used test data: {teacher}")
    signature_frame = pd.DataFrame(baseline_signatures)
    comparison_fields = [
        column
        for column in signature_frame.columns
        if column not in {"teacher_arm", "seed"}
    ]
    for seed, group in signature_frame.groupby("seed"):
        if any(group[field].nunique(dropna=False) != 1 for field in comparison_fields):
            raise RuntimeError(
                f"fresh matched baselines differ in training-affecting fields at seed {seed}"
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            ["teacher_arm", "teacher_id", "teacher_standalone_mean_R1"],
            as_index=False,
        )
        .agg(
            baseline_mean_R1=("baseline_mean_R1", "mean"),
            distilled_mean_R1=("distilled_mean_R1", "mean"),
            distillation_gain_mean_pp=("distillation_gain_pp", "mean"),
            distillation_gain_sd_pp=("distillation_gain_pp", "std"),
            distillation_gain_min_pp=("distillation_gain_pp", "min"),
            distillation_gain_max_pp=("distillation_gain_pp", "max"),
        )
        .sort_values("teacher_standalone_mean_R1")
    )
    strength_order = summary.sort_values("teacher_standalone_mean_R1")[
        "teacher_arm"
    ].tolist()
    utility_order = summary.sort_values(
        "distillation_gain_mean_pp", ascending=False
    )["teacher_arm"].tolist()
    destination = ROOT / "results/alignment_wave1/teacher_comparison"
    atomic_csv(frame, destination / "per_seed.csv")
    atomic_csv(summary, destination / "summary.csv")
    atomic_csv(signature_frame, destination / "baseline_semantic_audit.csv")
    result = {
        "status": "COMPLETE",
        "teachers": 3,
        "fresh_matched_baseline_per_teacher": True,
        "baseline_training_semantics_identical": True,
        "teacher_strength_order_weak_to_strong": strength_order,
        "distillation_utility_order_high_to_low": utility_order,
        "monotonic_strength_utility_order": strength_order
        == list(reversed(utility_order)),
        "interpretation_scope": (
            "Three heterogeneous teachers test whether standalone retrieval "
            "strength monotonically predicts utility for this fixed student. "
            "They do not identify a causal compactness, architecture, data, "
            "or distillation-heritage mechanism."
        ),
        "decision_rule": "DESCRIPTIVE_ONLY_NO_POSTHOC_THRESHOLD",
        "test_split_used": False,
    }
    atomic_json(result, destination / "report.json")
    return result


def component_smoke(pipeline_path: str | Path) -> dict[str, Any]:
    # exercise each optional component alone on real data before combinations
    _, pipeline = load_pipeline(pipeline_path)
    if not _gate_passed(pipeline, "distillation"):
        return {"status": "SKIPPED_GATE", "gate": "distillation"}
    device = get_device("auto")
    if device.type != "cuda":
        raise RuntimeError("component smoke test requires CUDA")
    recipes = [
        next(value for value in pipeline["ablations"]["recipes"] if value["id"] == name)
        for name in ("adapter", "lora", "distillation")
    ]
    results = []
    for recipe in recipes:
        job = {
            **recipe,
            "experiment_id": f"smoke_{recipe['id']}",
            "run_id": f"smoke_{recipe['id']}__seed_42",
            "seed": 42,
        }
        config = build_job_config(pipeline, job, "ablation")
        config["training"]["batch_size"] = 8
        config["data"]["num_workers"] = 0
        config["data"]["persistent_workers"] = False
        train_loader, _ = build_dataloaders(config)
        batch = next(iter(train_loader))
        model = build_model(config).to(device).train()
        parameters = model.parameter_summary()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        teacher = (
            TeacherCache(ROOT / str(config["distillation"]["cache_path"]))
            if recipe["distillation"]
            else None
        )
        losses = []
        for _ in range(3):
            images = batch["images"].to(device)
            captions = [str(value) for value in batch["captions"]]
            image_ids = [str(value) for value in batch["image_paths"]]
            text_ids = [str(value) for value in batch["text_image_paths"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images, captions)
                loss = multi_positive_contrastive_loss(
                    outputs["logits"], image_ids, text_ids
                )
                if teacher is not None:
                    teacher_image, teacher_text = teacher.lookup(
                        image_ids, text_ids, captions, device
                    )
                    auxiliary, _ = distillation_loss(
                        outputs,
                        teacher_image,
                        teacher_text,
                        teacher_scale=float(teacher.metadata["logit_scale"]),
                        temperature=float(config["distillation"]["temperature"]),
                        image_cosine_weight=float(config["distillation"]["image_cosine_weight"]),
                        text_cosine_weight=float(config["distillation"]["text_cosine_weight"]),
                        kl_weight=float(config["distillation"]["kl_weight"]),
                    )
                    loss = loss + float(config["distillation"]["strength"]) * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{recipe['id']} smoke loss is non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        inference_config = deep_update(config, {"recipe": {"distillation": False}})
        clone = build_model(inference_config)
        clone.load_state_dict(model.inference_state_dict(), strict=True)
        results.append(
            {
                "recipe": recipe["id"],
                "losses": losses,
                "loss_decreased": losses[-1] < losses[0],
                "parameter_summary": parameters,
                "lora_targets": model.lora_targets,
                "strict_inference_reload": True,
            }
        )
        del clone, model, optimizer, teacher
        gc.collect()
        torch.cuda.empty_cache()
    result = {"status": "COMPLETE", "components": results}
    atomic_json(result, output_root(pipeline) / "component_smoke.json")
    return result


def select_recipe(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    distillation = json.loads((output_root(pipeline) / "selection/distillation.json").read_text())
    if not distillation.get("proceed"):
        result = {"status": "SKIPPED_GATE", "gate": "distillation"}
        atomic_json(result, output_root(pipeline) / "selection/recipe.json")
        return result
    rows = []
    for job in ablation_jobs(pipeline):
        path = output_root(pipeline) / "ablation" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing ablation result {path}")
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("experiment_id", as_index=False).agg(
        mean_R1=("mean_R@1", "mean"),
        latency=("bidirectional_pair_latency_ms", "mean"),
        total_params=("params_total_inference", "mean"),
        trainable_params=("params_trainable_inference", "mean"),
    )
    base = load_config(ROOT / str(pipeline["base_config"]))
    openclip_path = (
        output_root(pipeline)
        / "references/openclip_vit_b32_quickgelu_openai/metrics.json"
    )
    openclip_latency = float(base["budgets"]["max_latency_ms"])
    if openclip_path.is_file():
        openclip_latency = float(
            json.loads(openclip_path.read_text())["bidirectional_pair_latency_ms"]
        )
    eligible = summary[
        (summary["total_params"] <= int(base["budgets"]["max_total_params"]))
        & (summary["trainable_params"] <= int(base["budgets"]["max_trainable_params"]))
        & (summary["latency"] <= openclip_latency)
    ]
    budget_failure = eligible.empty
    candidates = summary if budget_failure else eligible
    winner = candidates.sort_values("mean_R1", ascending=False).iloc[0]
    recipe = next(
        dict(value) for value in pipeline["ablations"]["recipes"] if value["id"] == winner["experiment_id"]
    )
    result = {
        "status": "SELECTED_BUDGET_FAILURE" if budget_failure else "SELECTED",
        "budget_failure": budget_failure,
        "experiment_id": winner["experiment_id"],
        "recipe": recipe,
        "mean_R@1": float(winner["mean_R1"]),
        "distillation_strength": _selected_strength(pipeline),
        "measured_openclip_latency_limit_ms": openclip_latency,
    }
    atomic_json(result, output_root(pipeline) / "selection/recipe.json")
    return result


def select_recovery_recipe(pipeline_path: str | Path) -> dict[str, Any]:
    # select a compact recovery recipe only when both seeds clear the gain floor
    _, pipeline = load_pipeline(pipeline_path)
    if not _direct_recovery_recipe_study(pipeline):
        raise RuntimeError("direct recovery recipe study is not enabled")
    if not _gate_passed(pipeline, "pair"):
        result = {"status": "SKIPPED_GATE", "gate": "pair"}
        atomic_json(result, output_root(pipeline) / "selection/recipe.json")
        return result
    rows = []
    for job in ablation_jobs(pipeline):
        path = output_root(pipeline) / "ablation" / job["run_id"] / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing recovery recipe result {path}")
        rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    baseline = frame[frame["experiment_id"] == "baseline"][["seed", "mean_R@1"]].rename(
        columns={"mean_R@1": "baseline_R1"}
    )
    if baseline["seed"].nunique() != len(pipeline["ablations"]["seeds"]):
        raise RuntimeError("matched recovery baseline is incomplete")
    candidates = frame[frame["experiment_id"] != "baseline"].merge(
        baseline, on="seed", validate="many_to_one"
    )
    candidates["paired_gain"] = candidates["mean_R@1"] - candidates["baseline_R1"]
    summary = (
        candidates.groupby("experiment_id", as_index=False)
        .agg(
            mean_R1=("mean_R@1", "mean"),
            mean_gain=("paired_gain", "mean"),
            min_seed_gain=("paired_gain", "min"),
            latency=("bidirectional_pair_latency_ms", "mean"),
            total_params=("params_total_inference", "mean"),
            trainable_params=("params_trainable_inference", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    base = load_config(ROOT / str(pipeline["base_config"]))
    openclip_path = (
        output_root(pipeline)
        / "references/openclip_vit_b32_quickgelu_openai/metrics.json"
    )
    latency_limit = float(base["budgets"]["max_latency_ms"])
    if openclip_path.is_file():
        latency_limit = float(
            json.loads(openclip_path.read_text())["bidirectional_pair_latency_ms"]
        )
    eligible = summary[
        (summary["total_params"] <= int(base["budgets"]["max_total_params"]))
        & (summary["trainable_params"] <= int(base["budgets"]["max_trainable_params"]))
        & (summary["latency"] <= latency_limit)
    ].copy()
    gain_floor = float(
        pipeline["selection"]["recipe_minimum_gain_over_matched_baseline"]
    )
    if eligible.empty:
        result = {
            "status": "STOPPED_BUDGET_GATE",
            "proceed": False,
            "gain_floor": gain_floor,
            "measured_openclip_latency_limit_ms": latency_limit,
        }
        atomic_json(result, output_root(pipeline) / "selection/recipe.json")
        atomic_csv(summary, output_root(pipeline) / "selection/recipe_summary.csv")
        return result
    winner = eligible.sort_values(
        ["mean_R1", "latency"], ascending=[False, True]
    ).iloc[0]
    proceed = (
        int(winner["seed_count"]) == len(pipeline["ablations"]["seeds"])
        and float(winner["min_seed_gain"]) >= gain_floor
    )
    recipe = next(
        dict(value)
        for value in pipeline["ablations"]["recipes"]
        if value["id"] == winner["experiment_id"]
    )
    result = {
        "status": "SELECTED" if proceed else "STOPPED_AT_GATE",
        "proceed": proceed,
        "experiment_id": str(winner["experiment_id"]),
        "recipe": recipe,
        "mean_R@1": float(winner["mean_R1"]),
        "mean_gain_over_matched_baseline": float(winner["mean_gain"]),
        "minimum_seed_gain_over_matched_baseline": float(winner["min_seed_gain"]),
        "required_gain_each_seed": gain_floor,
        "measured_openclip_latency_limit_ms": latency_limit,
        "distillation_strength": _selected_strength(pipeline),
    }
    atomic_json(result, output_root(pipeline) / "selection/recipe.json")
    atomic_csv(summary, output_root(pipeline) / "selection/recipe_summary.csv")
    return result


def select_wave0_lr_sweep(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rows: list[dict[str, Any]] = []
    for job in wave0_lr_calibration_jobs(pipeline):
        stage = str(job["artifact_stage"])
        path = output_root(pipeline) / stage / str(job["run_id"]) / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing Wave 0 LR calibration result {path}")
        rows.append({**job, **json.loads(path.read_text())})
    frame = pd.DataFrame(rows)
    per_batch: dict[str, Any] = {}
    for batch_size, group in frame.groupby("batch_size"):
        best = float(group["mean_R@1"].max())
        winners = group[group["mean_R@1"] == best]
        if len(winners) != 1:
            raise RuntimeError(
                f"exact LR-sweep tie at batch {batch_size}; no tie rule was supplied"
            )
        winner = winners.iloc[0]
        per_batch[str(int(batch_size))] = {
            "sweep_multiplier": float(winner["sweep_multiplier"]),
            "mean_R@1": best,
            "run_id": str(winner["run_id"]),
        }
    result = {"status": "SELECTED", "per_batch": per_batch}
    atomic_json(result, output_root(pipeline) / "selection/lr_sweep.json")
    atomic_csv(frame, output_root(pipeline) / "selection/lr_sweep_results.csv")
    return result


def _wave0_lr_source_job(
    pipeline: dict[str, Any], screen_job: dict[str, Any]
) -> dict[str, Any]:
    selection = json.loads(
        (output_root(pipeline) / "selection/lr_sweep.json").read_text()
    )
    source_run_id = str(
        selection["per_batch"][str(int(screen_job["batch_size"]))]["run_id"]
    )
    return next(
        job
        for job in wave0_lr_calibration_jobs(pipeline)
        if job["run_id"] == source_run_id
    )


def _reuse_wave0_lr_evaluation(
    pipeline: dict[str, Any], screen_job: dict[str, Any]
) -> dict[str, Any]:
    source_job = _wave0_lr_source_job(pipeline, screen_job)
    source_stage = str(source_job["artifact_stage"])
    source_path = (
        output_root(pipeline)
        / source_stage
        / str(source_job["run_id"])
        / "metrics.json"
    )
    if not source_path.is_file():
        raise RuntimeError(f"missing selected LR calibration metric {source_path}")
    config = build_job_config(pipeline, screen_job, "wave0-screen")
    fingerprint = _fingerprint(config)
    destination = (
        output_root(pipeline) / "wave0-screen" / str(screen_job["run_id"])
    )
    row = {
        **json.loads(source_path.read_text()),
        "status": "COMPLETE",
        "kind": "wave0-screen",
        "experiment_id": str(screen_job["experiment_id"]),
        "run_id": str(screen_job["run_id"]),
        "fingerprint_digest": fingerprint.digest,
        "reused_lr_calibration_run_id": str(source_job["run_id"]),
        "reused_without_retraining": True,
    }
    atomic_json(row, destination / "metrics.json")
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    write_fingerprint(destination / "fingerprint.json", fingerprint)
    source_config = build_job_config(pipeline, source_job, source_stage)
    checkpoint = ROOT / str(source_config["training"]["save_dir"]) / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    append_ledger_entry(
        pipeline,
        job=screen_job,
        stage="wave0-screen",
        status="COMPLETE",
        dev_r1=float(row["mean_R@1"]),
        test_r1=None,
        training_metrics=dict(saved.get("metrics", {})),
    )
    return row


def _wave0_recipe_from_job(
    pipeline: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    config = build_job_config(pipeline, job, "wave0-screen")
    return {
        "adapter": False,
        "distillation": False,
        "lora": False,
        "loss_type": str(job["loss_type"]),
        "captions_per_image": job.get("captions_per_image"),
        "memory_queue_size": int(job["memory_queue_size"]),
        "batch_size": int(job["batch_size"]),
        "sweep_multiplier": float(config["training"]["sweep_multiplier"]),
    }


def select_wave0_finalists(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    jobs = wave0_jobs(pipeline)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        path = output_root(pipeline) / "wave0-screen" / str(job["run_id"]) / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing Wave 0 screening result {path}")
        rows.append({**job, **json.loads(path.read_text())})
    frame = pd.DataFrame(rows)
    anchor = frame[
        (frame["loss_type"] == "infonce_queue")
        & (frame["captions_per_image"] == 2)
        & (frame["batch_size"] == 2048)
    ]
    if len(anchor) != 1:
        raise RuntimeError("Wave 0 replication anchor is not unique")
    anchor_r1 = float(anchor.iloc[0]["mean_R@1"])
    margin = anchor_r1 - 0.1702
    absolute_margin = abs(margin)
    support = (
        "STRONG"
        if absolute_margin <= 0.01
        else "QUALIFIED"
        if absolute_margin <= 0.02
        else "STOP"
    )
    spread = {
        "min": float(frame["mean_R@1"].min()),
        "max": float(frame["mean_R@1"].max()),
        "std": float(frame["mean_R@1"].std(ddof=1)),
    }
    batch_means = frame.groupby("batch_size")["mean_R@1"].mean()
    batch_effect = float(batch_means.loc[2048] - batch_means.loc[512])
    predictions = _wave0_predictions(pipeline)
    batch_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wave": "wave0",
        "question": "batch_effect",
        "prediction": predictions["batch_effect"]["statement"],
        "config_fingerprint": hash_payload(
            {
                "screening_fingerprints": sorted(
                    str(value) for value in frame["fingerprint_digest"]
                )
            }
        ),
        "seed": 42,
        "git_sha": code_fingerprint(ROOT, CODE_PATHS),
        "git_sha_is_code_fingerprint": True,
        "batch_size": None,
        "unique_images_per_batch": None,
        "text_rows_per_batch": None,
        "loss_type": "aggregate",
        "captions_per_image": "aggregate",
        "base_lr": None,
        "sqrt_scale_factor": None,
        "sweep_multiplier": None,
        "resolved_lr": None,
        "lr_scale_cap_hit": None,
        "mean_loss_magnitude": None,
        "dev_r1": batch_effect,
        "test_r1": None,
        "status": "COMPLETE",
    }
    ledger = ROOT / "logs/run_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(batch_entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    common = {
        "replication_anchor_R@1": anchor_r1,
        "replication_anchor_target_R@1": 0.1702,
        "replication_anchor_margin": margin,
        "replication_anchor_absolute_margin": absolute_margin,
        "replication_support": support,
        "screening_spread": spread,
        "batch_effect_2048_minus_512": batch_effect,
    }
    if support == "STOP":
        result = {
            "status": "STOPPED_REPLICATION_ANCHOR",
            "proceed": False,
            **common,
        }
        atomic_json(result, output_root(pipeline) / "selection/finalists.json")
        atomic_json(result, output_root(pipeline) / "selection/recipe.json")
        return result
    ranked = frame.sort_values("mean_R@1", ascending=False).head(3)
    finalists = []
    job_by_id = {str(job["experiment_id"]): job for job in jobs}
    for row in ranked.to_dict("records"):
        job = job_by_id[str(row["experiment_id"])]
        finalists.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "seed_42_R@1": float(row["mean_R@1"]),
                "recipe": _wave0_recipe_from_job(pipeline, job),
            }
        )
    result = {
        "status": "FINALISTS_SELECTED",
        "proceed": True,
        **common,
        "finalists": finalists,
    }
    atomic_json(result, output_root(pipeline) / "selection/finalists.json")
    for job in wave0_selection_jobs(pipeline):
        append_ledger_entry(
            pipeline, job=job, stage="wave0-select", status="PENDING"
        )
    atomic_csv(frame, output_root(pipeline) / "selection/screening_results.csv")
    return result


def _wave0_tiebreak_key(recipe: dict[str, Any]) -> tuple[int, int, int]:
    return (
        {"infonce_queue": 0, "infonce_no_queue": 1, "sigmoid": 2}[
            str(recipe["loss_type"])
        ],
        0 if recipe.get("captions_per_image") == 2 else 1,
        {512: 0, 1024: 1, 2048: 2}[int(recipe["batch_size"])],
    )


def select_wave0_recipe(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    finalists_path = output_root(pipeline) / "selection/finalists.json"
    finalists_payload = json.loads(finalists_path.read_text())
    if not finalists_payload.get("proceed"):
        return finalists_payload
    summaries: list[dict[str, Any]] = []
    for finalist in finalists_payload["finalists"]:
        values_3 = [float(finalist["seed_42_R@1"])]
        values_2: list[float] = []
        for seed in pipeline["wave0"]["finalist_seeds"]:
            run_id = f"{finalist['experiment_id']}__seed_{int(seed)}"
            path = output_root(pipeline) / "wave0-select" / run_id / "metrics.json"
            if not path.is_file():
                raise RuntimeError(f"missing Wave 0 finalist result {path}")
            value = float(json.loads(path.read_text())["mean_R@1"])
            values_3.append(value)
            values_2.append(value)
        summaries.append(
            {
                **finalist,
                "seed_values": values_3,
                "mean_3seed": sum(values_3) / 3,
                "mean_2seed": sum(values_2) / 2,
            }
        )
    rank_3 = sorted(summaries, key=lambda value: value["mean_3seed"], reverse=True)
    rank_2 = sorted(summaries, key=lambda value: value["mean_2seed"], reverse=True)
    best = float(rank_3[0]["mean_3seed"])
    equivalent = [
        value for value in rank_3 if best - float(value["mean_3seed"]) < 0.005
    ]
    winner = min(equivalent, key=lambda value: _wave0_tiebreak_key(value["recipe"]))
    result = {
        "status": "SELECTED_WAVE0_RECIPE",
        "proceed": True,
        "recipe": winner["recipe"],
        "experiment_id": winner["experiment_id"],
        "mean_3seed": winner["mean_3seed"],
        "mean_2seed": winner["mean_2seed"],
        "tie_break_triggered": len(equivalent) > 1,
        "rank_3seed": [value["experiment_id"] for value in rank_3],
        "rank_2seed": [value["experiment_id"] for value in rank_2],
        "winner_curse_ranking_disagreement": (
            rank_3[0]["experiment_id"] != rank_2[0]["experiment_id"]
        ),
        "finalists": summaries,
        **{
            key: finalists_payload[key]
            for key in (
                "replication_anchor_R@1",
                "replication_anchor_target_R@1",
                "replication_anchor_margin",
                "replication_anchor_absolute_margin",
                "replication_support",
                "screening_spread",
                "batch_effect_2048_minus_512",
            )
        },
    }
    atomic_json(result, output_root(pipeline) / "selection/recipe.json")
    return result


def lock_wave0_corrected_recipe(pipeline_path: str | Path) -> dict[str, Any]:
    # lock queue-free InfoNCE only after its two confirmation seeds evaluate
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    seed_jobs = [
        wave0_lr_control_jobs(pipeline)[0],
        *wave0_winner_confirmation_jobs(pipeline),
    ]
    values: list[float] = []
    for position, job in enumerate(seed_jobs):
        stage = "wave0-lr-control" if position == 0 else "wave0-winner-confirmation"
        path = output / stage / str(job["run_id"]) / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing evaluated winner-confirmation result {path}")
        values.append(float(json.loads(path.read_text())["mean_R@1"]))
    previous_path = output / "selection/recipe.json"
    previous = json.loads(previous_path.read_text())
    sigmoid_mean = float(
        previous.get("sigmoid_mean_3seed", previous["mean_3seed"])
    )
    corrected_mean = float(sum(values) / len(values))
    if corrected_mean <= sigmoid_mean:
        result = {
            **previous,
            "status": "CONFIRMATION_FAILED_NO_LOCK",
            "proceed": False,
            "corrected_candidate_mean_3seed": corrected_mean,
            "corrected_candidate_seed_values": values,
            "sigmoid_mean_3seed": sigmoid_mean,
            "wave1_unblocked": False,
        }
        atomic_json(result, previous_path)
        return result
    recipe = {
        "adapter": False,
        "distillation": False,
        "lora": False,
        "loss_type": "infonce_no_queue",
        "captions_per_image": None,
        "memory_queue_size": 0,
        "batch_size": 1024,
        "sweep_multiplier": 3.0,
    }
    mean_2seed = float(sum(values[1:]) / 2)
    historical_screen = {
        key: previous[key]
        for key in (
            "replication_anchor_R@1",
            "replication_anchor_target_R@1",
            "replication_anchor_margin",
            "replication_anchor_absolute_margin",
            "replication_support",
            "screening_spread",
            "batch_effect_2048_minus_512",
        )
        if key in previous
    }
    margin = float(pipeline["selection"]["practical_equivalence_margin"])
    result = {
        "status": "SELECTED_WAVE0_RECIPE",
        "proceed": True,
        "recipe": recipe,
        "experiment_id": "infonce_no_queue__captions_all__b1024__lr_3p0",
        "mean_3seed": corrected_mean,
        "mean_2seed": mean_2seed,
        "mean_2seed_seed_order": [43, 44],
        "mean_2seed_semantics": (
            "Confirmation mean over seeds 43 and 44 only. This is not a "
            "winner's-curse correction: the corrected queue-free configuration "
            "was not selected from the original 18-cell screening ranking."
        ),
        "winner_curse_correction_applicable": False,
        "seed_values": values,
        "seed_order": [42, 43, 44],
        "sigmoid_mean_3seed": sigmoid_mean,
        "corrected_minus_sigmoid_mean_3seed": corrected_mean - sigmoid_mean,
        "practical_equivalence_margin": margin,
        "winner_on_mean_3seed": True,
        "tie_break_triggered": abs(corrected_mean - sigmoid_mean) <= margin,
        "tie_break_rule": "fewer_changes_from_v4_recipe",
        "tie_break_outcome_if_needed": "infonce_no_queue",
        "selection_basis": (
            "Higher three-seed mean; the difference exceeded the practical "
            "equivalence margin, so the tie-break was not needed."
        ),
        "historical_wave0_screen": historical_screen,
        "prior_selection_artifact": "selection/finalists.json",
        "prior_selection_superseded": True,
        "locked_student": "dinov3_vits16__all_minilm_l6_v2",
        "student_pair_changed": False,
        "student_pair_decision": "KEEP_MINILM_FOR_MAIN_STUDY",
        "student_pair_decision_permanent_for_study": True,
        "student_pair_decision_rationale": [
            (
                "The queue study, 18-cell screen, dose-response curve, v4 "
                "teacher caches, and all 31 claim-bearing runs use MiniLM; "
                "switching would disconnect the factorial from the pair on "
                "which the measured queue curve was established."
            ),
            (
                "E5's 1.56pp corrected-recipe gain carries no headline claim: "
                "it does not reach OpenCLIP, alter the queue mechanism, or "
                "change the pair-ranking finding."
            ),
            (
                "E5 is retained as evidence that the recipe correction "
                "generalises across text encoders and that the queue effect is "
                "not a MiniLM-specific artifact."
            ),
        ],
        "deferred_best_available_configuration": {
            "pair": "dinov3_vits16__e5_small_v2",
            "purpose": "efficiency_frontier_plot_only",
            "timing": "end_of_project",
            "part_of_main_study": False,
            "run_now": False,
        },
        "loss_family_finding": False,
        "dominant_finding": "memory_queue_staleness",
        "wave1_unblocked": True,
    }
    atomic_json(result, previous_path)
    return result


def evaluate_transfer(pipeline_path: str | Path, index: int | None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(transfer_jobs(pipeline), index)
    csv_path = ROOT / str(pipeline["optional_transfer"]["flickr30k_csv"])
    if not csv_path.is_file():
        return {
            "status": "SKIPPED_OPTIONAL_MISSING",
            "dataset": "flickr30k",
            "path": str(csv_path),
            "run_id": job["run_id"],
        }
    if job["kind"] == "final" and not _gate_passed(pipeline, "recipe"):
        return {"status": "SKIPPED_GATE", "gate": "recipe", "run_id": job["run_id"]}
    destination = output_root(pipeline) / "transfer" / str(job["run_id"])
    device = get_device("auto")
    base = load_config(ROOT / str(pipeline["base_config"]))
    transfer_fingerprint = Fingerprint(
        config_hash=hash_config({"job": job, "evaluation": base["evaluation"]}),
        dataset_split_hash=sha256_file(csv_path),
        model_checkpoint_id=str(job.get("reference_id", job.get("final_run_id"))),
        cache_version="flickr30k-transfer-v1",
        preprocessing_hash=hash_payload({"kind": job["kind"]}),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
        seed=int(job["seed"]) if job.get("seed") is not None else None,
    )
    existing = destination / "metrics.json"
    if existing.is_file() and is_fresh(destination / "fingerprint.json", transfer_fingerprint):
        return {**json.loads(existing.read_text()), "status": "SKIPPED_FRESH"}
    if job["kind"] == "reference":
        model = build_reference(str(job["reference_id"])).to(device)
        loader = native_loader(
            model,
            csv_path,
            image_root=ROOT,
            batch_size=int(base["evaluation"].get("batch_size", 256)),
            num_workers=int(base["data"]["num_workers"]),
        )
        parameters = {
            "params_total_inference": sum(parameter.numel() for parameter in model.parameters()),
            "params_trainable_inference": 0,
        }
    else:
        final_job = next(value for value in final_jobs(pipeline) if value["run_id"] == job["final_run_id"])
        config = build_job_config(pipeline, final_job, "final")
        model = build_model(config).to(device)
        fingerprint = _fingerprint(config, full_train=True)
        checkpoint = ROOT / str(config["training"]["save_dir"]) / "best.pt"
        load_training_checkpoint(checkpoint, model, device=device, expected_fingerprint=fingerprint)
        transfer_config = deep_update(
            config,
            {
                "data": {"val_csv": str(csv_path)},
                "training": {"batch_size": int(config["evaluation"].get("batch_size", 256))},
            },
        )
        _, loader = build_dataloaders(transfer_config)
        parameters = model.parameter_summary()
    embeddings = extract_embeddings(model, loader, device)
    metrics = _metrics_from_embeddings(embeddings, list(base["evaluation"]["k_values"]))
    row = {
        "status": "COMPLETE",
        "source": "measured_local",
        "kind": "transfer_flickr30k",
        "experiment_id": job["experiment_id"],
        "run_id": job["run_id"],
        "seed": job.get("seed"),
        **metrics,
        **parameters,
        "dataset_sha256": sha256_file(csv_path),
    }
    atomic_json(row, destination / "metrics.json")
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    write_fingerprint(destination / "fingerprint.json", transfer_fingerprint)
    return row


def oracle(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    rankings = []
    labels = []
    # Complementarity is assessed across the six distinct frozen expert pairs
    # on the development split. Seeds of one model are not distinct experts and
    # would make a misleading router oracle.
    for job in pair_jobs(pipeline):
        path = output_root(pipeline) / "pair" / job["run_id"] / "rankings.pt"
        if path.is_file():
            rankings.append(torch.load(path, map_location="cpu", weights_only=False))
            labels.append(job["run_id"])
    if len(rankings) < 2:
        result = {"status": "INCOMPLETE", "reason": "fewer than two expert-pair ranking files"}
        atomic_json(result, output_root(pipeline) / "oracle.json")
        return result
    image_paths = rankings[0]["image_paths"]
    text_paths = rankings[0]["text_image_paths"]
    correctness = []
    for ranking in rankings:
        i2t_top = ranking["i2t_indices"][:, 0].tolist()
        t2i_top = ranking["t2i_indices"][:, 0].tolist()
        i2t_correct = torch.tensor(
            [text_paths[index] == image_paths[row] for row, index in enumerate(i2t_top)]
        )
        t2i_correct = torch.tensor(
            [image_paths[index] == text_paths[row] for row, index in enumerate(t2i_top)]
        )
        correctness.append(torch.cat((i2t_correct, t2i_correct)))
    individual = [float(value.float().mean()) for value in correctness]
    oracle_value = float(torch.stack(correctness).any(dim=0).float().mean())
    final_rows = [
        json.loads(path.read_text())
        for path in sorted((output_root(pipeline) / "final").glob("*/metrics.json"))
    ]
    final_mean = (
        float(pd.DataFrame(final_rows)["mean_R@1"].mean()) if final_rows else float("nan")
    )
    openclip_path = (
        output_root(pipeline)
        / "references/openclip_vit_b32_quickgelu_openai/metrics.json"
    )
    openclip = json.loads(openclip_path.read_text()) if openclip_path.is_file() else {}
    openclip_mean = float(openclip.get("mean_R@1", float("nan")))
    fraction = final_mean / openclip_mean if openclip_mean > 0 else float("nan")
    selection = pipeline["selection"]
    unique_gain = oracle_value - max(individual)
    router_recommended = (
        unique_gain >= float(selection["router_minimum_unique_oracle_gain"])
        and fraction >= float(selection["router_minimum_openclip_fraction"])
    )
    result = {
        "status": "COMPLETE",
        "models": labels,
        "individual_bidirectional_R@1": individual,
        "oracle_bidirectional_R@1": oracle_value,
        "unique_oracle_gain": unique_gain,
        "final_openclip_fraction": fraction,
        "router_recommended": router_recommended,
        "note": "Ground-truth oracle upper bound on the development split; not a deployable router result.",
    }
    atomic_json(result, output_root(pipeline) / "oracle.json")
    return result


def report(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    if _wave0_mode(pipeline):
        return report_wave0(pipeline_path)
    output = output_root(pipeline)
    rows = []
    for kind in (
        "references", "pair", "confirmation", "sensitivity", "ablation", "final", "transfer"
    ):
        for path in sorted((output / kind).glob("*/metrics.json")):
            rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    if frame.empty:
        result = {"status": "INCOMPLETE", "rows": 0}
        atomic_json(result, output / "report.json")
        return result
    atomic_csv(frame, output / "measured_results.csv")
    numeric = [
        value for value in (
            "i2t_R@1", "t2i_R@1", "mean_R@1", "bidirectional_pair_latency_ms",
            "params_total_inference", "params_trainable_inference"
        ) if value in frame
    ]
    summary = frame.groupby(["kind", "experiment_id"], dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["__".join(value).strip("_") if isinstance(value, tuple) else value for value in summary.columns]
    atomic_csv(summary, output / "measured_summary.csv")
    if "mean_R@1" in frame:
        seed_statistics = (
            frame.groupby(["kind", "experiment_id"], dropna=False)["mean_R@1"]
            .agg(["mean", "std", "count", "min", "max"])
            .reset_index()
        )
        critical = {2: 12.706, 3: 4.303}
        seed_statistics["ci95_halfwidth"] = seed_statistics.apply(
            lambda row: (
                critical.get(int(row["count"]), float("nan"))
                * float(row["std"])
                / math.sqrt(int(row["count"]))
                if int(row["count"]) in critical and pd.notna(row["std"])
                else float("nan")
            ),
            axis=1,
        )
        seed_statistics["ci_note"] = seed_statistics["count"].map(
            lambda count: "Student-t; unstable small-n interval" if count in (2, 3) else "not reported"
        )
        atomic_csv(seed_statistics, output / "seed_statistics.csv")
    literature = pd.DataFrame(pipeline.get("literature_context", []))
    atomic_csv(literature, output / "literature_context.csv")
    final_rows = len(list((output / "final").glob("*/metrics.json")))
    probe_payload: dict[str, Any] | None = None
    if _probe_mode(pipeline):
        probe_path = output / "probe_report.json"
        if probe_path.is_file():
            probe_payload = json.loads(probe_path.read_text())
        sensitivity_rows = len(list((output / "sensitivity").glob("*/metrics.json")))
        status = (
            "COMPLETE"
            if probe_payload is not None
            and probe_payload.get("status") == "COMPLETE"
            and sensitivity_rows == len(sensitivity_jobs(pipeline))
            else "INCOMPLETE"
        )
    else:
        status = (
            "COMPLETE" if final_rows == len(final_jobs(pipeline)) else "INCOMPLETE"
        )
    title = (
        "# Alignment v4 distillation capture probe"
        if _probe_mode(pipeline)
        else "# Alignment v3 results"
    )
    probe_section = (
        "\n\n## Capture probe decision\n\n```json\n"
        + json.dumps(probe_payload, indent=2)
        + "\n```\n"
        if probe_payload is not None
        else ""
    )
    markdown = (
        title
        + "\n\n"
        f"Status: **{status}**\n\n"
        "## Locally measured results\n\n"
        + frame.sort_values(["kind", "experiment_id", "seed"], na_position="first").to_markdown(index=False)
        + probe_section
        + "\n\n## Literature-only context\n\n"
        + (literature.to_markdown(index=False) if not literature.empty else "No literature rows configured.")
        + "\n\nSmall-seed confidence intervals use Student-t critical values and are labelled unstable.\n"
    )
    atomic_text(markdown, output / "report.md")
    result = {
        "status": status,
        "measured_rows": len(frame),
        "final_rows": final_rows,
        "probe_complete": bool(probe_payload is not None) if _probe_mode(pipeline) else None,
    }
    atomic_json(result, output / "report.json")
    return result


def report_wave0(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    recipe_path = output / "selection/recipe.json"
    recipe = json.loads(recipe_path.read_text()) if recipe_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    for stage in (
        "wave0-lrsweep",
        "wave0-screen",
        "wave0-select",
        "wave0-lr-control",
        "wave0-queue-ablation",
        "wave0-winner-confirmation",
        "wave0-rescreen",
        "wave0-rescreen-confirmation",
    ):
        for path in sorted((output / stage).glob("*/metrics.json")):
            rows.append(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        atomic_csv(frame, output / "measured_results.csv")
    predictions = _wave0_predictions(pipeline)
    prediction_rows: list[dict[str, Any]] = []
    if recipe:
        historical = dict(recipe.get("historical_wave0_screen", {}))
        anchor = float(
            historical.get(
                "replication_anchor_R@1",
                recipe.get("replication_anchor_R@1", float("nan")),
            )
        )
        winner = str(recipe.get("experiment_id", "not selected"))
        batch_effect = float(
            historical.get(
                "batch_effect_2048_minus_512",
                recipe.get("batch_effect_2048_minus_512", float("nan")),
            )
        )
        observed = {
            "replication_anchor": anchor,
            "wave0_winner": winner,
            "batch_effect": batch_effect,
        }
        hits = {
            "replication_anchor": 0.1502 <= anchor <= 0.1702,
            "wave0_winner": (
                winner.startswith("sigmoid__captions_all__b1024")
                or winner.startswith("sigmoid__captions_all__b2048")
            ),
            "batch_effect": 0 < batch_effect < 0.02,
        }
        for prediction_id in (
            "replication_anchor",
            "wave0_winner",
            "batch_effect",
        ):
            prediction_rows.append(
                {
                    "id": prediction_id,
                    "statement": predictions[prediction_id]["statement"],
                    "observed": observed[prediction_id],
                    "outcome": "hit" if hits[prediction_id] else "miss",
                    "if_wrong": (
                        ""
                        if hits[prediction_id]
                        else predictions[prediction_id]["if_wrong"]
                    ),
                }
            )
    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        atomic_csv(prediction_frame, output / "prediction_outcomes.csv")
    status = (
        "COMPLETE"
        if str(recipe.get("status", "")).startswith("SELECTED")
        else str(recipe.get("status", "INCOMPLETE"))
    )
    limitations = (
        "## Limitations\n\n"
        "- **Tie-break was not invoked:** queue-free InfoNCE exceeded sigmoid's "
        "three-seed mean by 0.797pp, beyond the pre-registered 0.5pp practical-"
        "equivalence margin.\n"
        "- **Historical LR asymmetry:** sigmoid received the original per-batch "
        "LR sweep. Subsequent matched-LR controls tested queue-free InfoNCE at "
        "x3 and x6 before recipe lock.\n"
        "- **Compounding sigmoid/captions-all limitation:** sigmoid multipliers "
        "are tuned at captions=2 and applied to captions=all, whose per-image "
        "normalization already increases loss magnitude with text-row count.\n\n"
        "The controls reduce but do not erase the fact that the recipe was "
        "selected on the development split.\n"
    )
    queue_audit = (
        "## Queue audit and revised finding\n\n"
        "- The dominant observed effect is the memory queue (~19pp at "
        "captions=all, batch=1024), not the loss family (~1pp before matched-LR "
        "controls).\n"
        "- Queue entries are detached, FIFO eviction at 16,384 entries is active, "
        "and same-image captions are masked as positives rather than false "
        "negatives.\n"
        "- Embeddings are unit-normalized when produced, but are enqueued after "
        "the optimizer step from the pre-update forward pass. They are stale at "
        "insertion and are never recomputed in the updated projection space; no "
        "momentum key encoder is used.\n"
        "- The batch effect is queue-conditional: +3.7/+4.6pp with the queue, "
        "-2.0/-2.2pp without it, and approximately flat for sigmoid. The v3/v4 "
        "gap is therefore better explained by queue staleness ratio than batch "
        "size alone: 16,384 historical image entries versus 512 current images "
        "is 32:1, versus 2,048 current images is 8:1.\n"
        "- Cell 0 (`infonce_queue`, captions=2, batch=512; 13.32%) reproduces the "
        "v4 baseline, and cell 2 (17.02%) reproduces v3.\n"
        "- Scope note for prior chapters: every reported v2, v3, and v4 result "
        "used the memory queue. Conclusions from those chapters therefore apply "
        "to the queued training recipe, not queue-free frozen alignment in "
        "general.\n\n"
        "This audit found no detach, eviction, or positive-mask implementation "
        "defect. It did find a methodological design mismatch between a changing "
        "student projector and historical detached keys. The staleness "
        "explanation is strongly consistent with the screen, but remains an "
        "inference rather than an isolated causal result.\n"
    )
    lr_bookkeeping = (
        "## Sigmoid LR provenance\n\n"
        "The three captions=2 calibration cells ran at multiplier x1.0, but "
        "their screening metrics were later replaced by aliases to the selected "
        "x3.0 sweep jobs. Their original screening checkpoint directories still "
        "contain the obsolete x1.0 checkpoints. The three captions=all screening "
        "cells and every seed-43/44 finalist rerun genuinely ran at x3.0. Thus "
        "all six sigmoid values used for finalist selection were x3.0 results, "
        "but the captions=2 screening artifact directories have mixed "
        "provenance.\n\n"
        "The LR multiplier is material, not noise: at captions=2, x1.0 produced "
        "33.12%/33.35%/30.70% for batches 512/1024/2048, while x3.0 produced "
        "35.65%/36.05%/36.20%. This is an approximately 3–5pp tuning effect and "
        "must qualify every loss-family comparison.\n\n"
        "Matched-LR control C confirmed queue-free InfoNCE at x3 (37.58%) "
        "exceeds the locked sigmoid candidate (36.61%). Its three-seed mean is "
        "37.32%, versus 36.52% for sigmoid. The difference exceeds the 0.5pp "
        "equivalence margin, so the tie-break was not required; it would also "
        "have favoured queue-free InfoNCE as the fewer-change recipe.\n"
    )
    hardware_split = (
        "## Hardware and precision split\n\n"
        "- Controls C/D and the accepted queue-capacity ablation tasks 0–6 from "
        "array 2223439 ran on the teaching partition's RTX PRO 6000 Blackwell "
        "GPUs with native BF16.\n"
        "- The two winner-confirmation seeds and all six pair re-screen jobs run "
        "on the teaching partition's RTX PRO 6000 Blackwell GPUs with native "
        "BF16, keeping the corrected-recipe comparison hardware matched.\n"
        "- No follow-up training is run in FP16.\n"
    )
    revised_result = (
        "## Corrected-recipe result\n\n"
        "- Queue dose response (training-time best dev mean bidirectional R@1): "
        "0 → 35.76%, 1,024 → 30.06%, 4,096 → 24.78%, 8,192 → 21.21%, "
        "16,384 → 16.77%.\n"
        "- Decomposition from the v4 baseline (13.32%) to the seed-42 corrected "
        "recipe (37.58%): queue removal ≈19.0pp, LR ×3 ≈1.8pp, all captions "
        "≈0.9pp, and batch 512→1024 ≈0.3pp.\n"
        "- The loss family is **not** the finding: sigmoid and queue-free "
        "InfoNCE differ by roughly 1pp, while the queue accounts for roughly "
        "19pp.\n"
        "- The seven recovered values have completed result-level evaluation. "
        "The anchor train/eval discrepancy was 0.00004pp.\n"
        "- The queue stores post-projection embeddings. A frozen backbone does "
        "not prevent staleness because the projector—the component defining "
        "that embedding space—is the component being updated.\n"
    )
    pre_registered = (
        "## Pre-registered interpretation\n\n"
        "If the replication anchor is within 1pp of 17.02%, batch size is the "
        "remaining explanation for the v3/v4 baseline gap without qualification. "
        "At 1–2pp, support is qualified and the residual is unexplained. Outside "
        "2pp, the wave stops.\n\n"
        "A higher locked baseline mechanically lowers capture fraction for fixed "
        "teacher performance and roughly fixed distillation gain because the "
        "capture denominator grows; this is algebra, not a new empirical result.\n\n"
        "The v3 batch 2048 value came from throughput probing; Wave 0 pins 2048 as "
        "a factorial level, so matching values are not an independent rediscovery.\n"
    )
    markdown = (
        "# Alignment v4 Wave 0 — recipe lock\n\n"
        f"Status: **{status}**\n\n"
        "## Locked recipe / stop record\n\n```json\n"
        + json.dumps(recipe, indent=2)
        + "\n```\n\n## Prediction outcomes\n\n"
        + (
            prediction_frame.to_markdown(index=False)
            if not prediction_frame.empty
            else "Not available until selection."
        )
        + "\n\n"
        + limitations
        + "\n"
        + queue_audit
        + "\n"
        + lr_bookkeeping
        + "\n"
        + hardware_split
        + "\n"
        + revised_result
        + "\n"
        + pre_registered
    )
    atomic_text(markdown, output / "report.md")
    result = {
        "status": status,
        "measured_rows": len(frame),
        "prediction_rows": len(prediction_frame),
        "recipe_control_status": pipeline["wave0"].get("lr_control_status"),
        "queue_audit_verdict": "NO_MECHANICAL_BUG_DESIGN_STALENESS_RISK",
        "prior_v2_v3_v4_scope": "MEMORY_QUEUE_ENABLED",
    }
    atomic_json(result, output / "report.json")
    return result


def report_wave0_queue_ablation(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    existing = {
        0: output / "wave0-screen/infonce_no_queue__captions_all__b1024__seed_42/metrics.json",
        16384: output / "wave0-screen/infonce_queue__captions_all__b1024__seed_42/metrics.json",
    }
    rows: list[dict[str, Any]] = []
    for queue_size, path in existing.items():
        if not path.is_file():
            raise RuntimeError(f"missing queue dose-response anchor {path}")
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "memory_queue_size": queue_size,
                "mean_R@1": float(metrics["mean_R@1"]),
                "run_id": str(metrics["run_id"]),
                "source": "existing_wave0_screen",
            }
        )
    for job in wave0_queue_ablation_jobs(pipeline):
        path = output / "wave0-queue-ablation" / str(job["run_id"]) / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"missing queue ablation result {path}")
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "memory_queue_size": int(job["memory_queue_size"]),
                "mean_R@1": float(metrics["mean_R@1"]),
                "run_id": str(job["run_id"]),
                "source": "new_queue_ablation",
            }
        )
    frame = pd.DataFrame(rows).sort_values("memory_queue_size")
    destination = output / "wave0-queue-ablation"
    atomic_csv(frame, destination / "dose_response.csv")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        frame["memory_queue_size"], frame["mean_R@1"] * 100.0,
        marker="o", linewidth=2,
    )
    axis.set_xlabel("Memory queue capacity")
    axis.set_ylabel("Mean bidirectional R@1 (%)")
    axis.set_title("Queue capacity dose response")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / "dose_response.png", dpi=180)
    plt.close(figure)
    result = {
        "status": "COMPLETE",
        "recipe_status": pipeline["wave0"]["lr_control_status"],
        "wave1_unblocked": False,
        "points": frame.to_dict(orient="records"),
    }
    atomic_json(result, destination / "report.json")
    atomic_text(
        "# Wave 0 queue-capacity dose response\n\n"
        "Recipe remains **PROVISIONAL_PENDING_LR_CONTROL**. This report does not "
        "select a recipe or unblock Wave 1.\n\n"
        + frame.to_markdown(index=False)
        + "\n",
        destination / "report.md",
    )
    return result


def report_wave0_2080ti_pilot(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = wave0_queue_ablation_jobs(pipeline)[0]
    checkpoint_dir = (
        checkpoint_root(pipeline) / "wave0-queue-ablation" / str(job["run_id"])
    )
    metrics_path = checkpoint_dir / "metrics.csv"
    config_path = checkpoint_dir / "config.yaml"
    if not metrics_path.is_file() or not config_path.is_file():
        raise RuntimeError("2080 Ti timing pilot has not completed training")
    config = load_config(config_path)
    if config["training"]["precision"] != "bf16":
        raise RuntimeError("2080 Ti timing pilot did not use BF16")
    if config.get("provenance", {}).get("slurm_partition_profile") != "2080ti":
        raise RuntimeError("timing pilot was not produced by the 2080ti profile")
    frame = pd.read_csv(metrics_path)
    if "epoch_seconds" not in frame or frame["epoch_seconds"].dropna().empty:
        raise RuntimeError("timing pilot did not record epoch_seconds")
    seconds = frame["epoch_seconds"].dropna().astype(float)
    conservative_ratio = float(seconds.mean()) / 61.0
    result = {
        "status": "WITHIN_4X" if conservative_ratio <= 4.0 else "OVER_4X_RECONSIDER",
        "run_id": str(job["run_id"]),
        "gpu_family": "RTX 2080 Ti",
        "bf16_mode": "emulated",
        "epochs_measured": int(len(seconds)),
        "mean_seconds_per_epoch": float(seconds.mean()),
        "median_seconds_per_epoch": float(seconds.median()),
        "minimum_seconds_per_epoch": float(seconds.min()),
        "maximum_seconds_per_epoch": float(seconds.max()),
        "teaching_reference_seconds_per_epoch": [56.0, 61.0],
        "conservative_ratio_vs_61_seconds": conservative_ratio,
        "within_four_x": conservative_ratio <= 4.0,
        "projected_12_epoch_hours": float(seconds.mean()) * 12.0 / 3600.0,
        "three_day_window_hours": 72.0,
    }
    destination = output_root(pipeline) / "wave0-queue-ablation/2080ti_pilot"
    atomic_json(result, destination / "report.json")
    atomic_text(
        "# RTX 2080 Ti emulated-BF16 timing pilot\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        destination / "report.md",
    )
    return result


def report_wave0_rescreen(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    current_rows: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    for job in wave0_rescreen_jobs(pipeline):
        current_path = output / "wave0-rescreen" / str(job["run_id"]) / "metrics.json"
        original_run_id = f"{job['experiment_id']}__seed_42"
        original_path = (
            ROOT / "results/alignment_v3/pair" / original_run_id / "metrics.json"
        )
        if not current_path.is_file():
            raise RuntimeError(f"missing Wave 0 re-screen result {current_path}")
        if not original_path.is_file():
            raise RuntimeError(f"missing original v3 pair result {original_path}")
        current_metrics = json.loads(current_path.read_text())
        original_metrics = json.loads(original_path.read_text())
        identity = {
            "experiment_id": str(job["experiment_id"]),
            "vision_encoder": str(job["vision_encoder"]),
            "text_encoder": str(job["text_encoder"]),
        }
        current_rows.append(
            {
                **identity,
                "dev_R@1": float(current_metrics["mean_R@1"]),
                "i2t_R@1": float(current_metrics["i2t_R@1"]),
                "t2i_R@1": float(current_metrics["t2i_R@1"]),
            }
        )
        original_rows.append(
            {
                **identity,
                "dev_R@1": float(original_metrics["mean_R@1"]),
                "i2t_R@1": float(original_metrics["i2t_R@1"]),
                "t2i_R@1": float(original_metrics["t2i_R@1"]),
            }
        )
    current = pd.DataFrame(current_rows).sort_values(
        "dev_R@1", ascending=False
    ).reset_index(drop=True)
    original = pd.DataFrame(original_rows).sort_values(
        "dev_R@1", ascending=False
    ).reset_index(drop=True)
    current["rank"] = current.index + 1
    original["rank"] = original.index + 1
    current_order = current["experiment_id"].tolist()
    original_order = original["experiment_id"].tolist()
    changed = current_order != original_order
    comparison = original.rename(
        columns={
            "rank": "v3_rank",
            "dev_R@1": "v3_dev_R@1",
            "i2t_R@1": "v3_i2t_R@1",
            "t2i_R@1": "v3_t2i_R@1",
        }
    ).merge(
        current.rename(
            columns={
                "rank": "rescreen_rank",
                "dev_R@1": "rescreen_dev_R@1",
                "i2t_R@1": "rescreen_i2t_R@1",
                "t2i_R@1": "rescreen_t2i_R@1",
            }
        ),
        on=["experiment_id", "vision_encoder", "text_encoder"],
    ).sort_values("rescreen_rank")
    vision_stable = all(
        float(
            frame[
                (frame["vision_encoder"] == "dinov3_vits16")
                & (frame["text_encoder"] == text)
            ]["dev_R@1"].iloc[0]
        )
        > float(
            frame[
                (frame["vision_encoder"] == "dinov3_convnext_tiny")
                & (frame["text_encoder"] == text)
            ]["dev_R@1"].iloc[0]
        )
        for frame in (original, current)
        for text in ("all_minilm_l6_v2", "e5_small_v2", "bge_small_en")
    )
    queued_text_orders = {
        vision: (
            original[original["vision_encoder"] == vision]
            .sort_values("dev_R@1", ascending=False)["text_encoder"]
            .tolist()
        )
        for vision in ("dinov3_convnext_tiny", "dinov3_vits16")
    }
    corrected_text_orders = {
        vision: (
            current[current["vision_encoder"] == vision]
            .sort_values("dev_R@1", ascending=False)["text_encoder"]
            .tolist()
        )
        for vision in ("dinov3_convnext_tiny", "dinov3_vits16")
    }
    directional_gain = comparison.assign(
        i2t_gain=lambda value: value["rescreen_i2t_R@1"] - value["v3_i2t_R@1"],
        t2i_gain=lambda value: value["rescreen_t2i_R@1"] - value["v3_t2i_R@1"],
    )
    mean_i2t_gain = float(directional_gain["i2t_gain"].mean())
    mean_t2i_gain = float(directional_gain["t2i_gain"].mean())
    queue_spread = float(original["dev_R@1"].max() - original["dev_R@1"].min())
    corrected_spread = float(current["dev_R@1"].max() - current["dev_R@1"].min())
    destination = output / "wave0-rescreen"
    atomic_csv(comparison, destination / "ranking_comparison.csv")
    result = {
        "status": "RANKING_CHANGED_STOP_FOR_DECISION" if changed else "RANKING_HELD",
        "recipe_status": pipeline["wave0"]["lr_control_status"],
        "wave1_unblocked": False,
        "ranking_changed": changed,
        "v3_winner": original_order[0],
        "corrected_recipe_winner": current_order[0],
        "v3_ranking": original_order,
        "corrected_recipe_ranking": current_order,
        "locked_student": "dinov3_vits16__all_minilm_l6_v2",
        "student_pair_changed": False,
        "pair_adoption_status": "DECIDED_KEEP_MINILM",
        "student_pair_decision_permanent_for_study": True,
        "vision_ranking_stable": vision_stable,
        "vision_ranking": "dinov3_vits16 > dinov3_convnext_tiny",
        "queued_text_rankings": queued_text_orders,
        "corrected_text_rankings": corrected_text_orders,
        "text_ranking_inverted": all(
            order == ["e5_small_v2", "bge_small_en", "all_minilm_l6_v2"]
            for order in corrected_text_orders.values()
        )
        and all(order[0] == "all_minilm_l6_v2" for order in queued_text_orders.values()),
        "modality_specific_confound": (
            "text_encoder_selection_changed_vision_encoder_selection_stable"
        ),
        "text_rows_per_image_row_ratio": 5.0,
        "mean_i2t_gain_after_queue_removal": mean_i2t_gain,
        "mean_t2i_gain_after_queue_removal": mean_t2i_gain,
        "i2t_extra_gain_over_t2i": mean_i2t_gain - mean_t2i_gain,
        "queued_six_pair_spread": queue_spread,
        "corrected_six_pair_spread": corrected_spread,
        "queue_spread_is_directionally_smaller": corrected_spread > queue_spread,
        "material_spread_compression_supported": False,
        "spread_note": (
            "The queue-era scores occupied a low 14.20–17.02% band and the "
            "corrected scores span 35.91–38.90%. Both max–min ranges are about "
            "3pp, so the exact data support severe level suppression and ranking "
            "flattening/inversion, not a materially narrower total max–min range."
        ),
    }
    atomic_json(result, destination / "report.json")
    atomic_text(
        "# Wave 0 v3-pair re-screen\n\n"
        f"Status: **{result['status']}**\n\n"
        "The locked student remains **DINOv3 ViT-S/16 + all-MiniLM-L6-v2**. "
        "The ranking change is a finding, but the main-study pair is permanently "
        "retained to preserve continuity with the MiniLM queue curve, screen, "
        "teacher caches, and 31 claim-bearing runs. Wave 1 remains blocked.\n\n"
        "## Structure of the ranking change\n\n"
        "- **Vision ranking is stable:** ViT-S/16 exceeds ConvNeXt-Tiny at every "
        "text encoder under both queued and corrected recipes.\n"
        "- **Text ranking inverted:** MiniLM leads within both vision families "
        "under the queue; after correction both are E5 > BGE > MiniLM.\n"
        "- This is modality-specific: text-encoder selection changed while "
        "vision-encoder selection remained intact.\n"
        "- The text side enqueues about 5× as many rows per step. Across these "
        f"six pairs, queue removal improved i2t by {mean_i2t_gain * 100:.2f}pp "
        f"versus {mean_t2i_gain * 100:.2f}pp for t2i, an extra "
        f"{(mean_i2t_gain - mean_t2i_gain) * 100:.2f}pp on i2t. Together with "
        "the text-ranking inversion, these are three observations pointing in "
        "the same direction.\n"
        f"- The queued band was {original['dev_R@1'].min() * 100:.2f}–"
        f"{original['dev_R@1'].max() * 100:.2f}% "
        f"({queue_spread * 100:.2f}pp); corrected is "
        f"{current['dev_R@1'].min() * 100:.2f}–"
        f"{current['dev_R@1'].max() * 100:.2f}% "
        f"({corrected_spread * 100:.2f}pp). Both total spreads are about 3pp; "
        "the defensible claim is suppression plus ranking flattening/inversion, "
        "not a materially smaller max–min range.\n\n"
        + comparison.to_markdown(index=False)
        + "\n",
        destination / "report.md",
    )
    return result


def report_wave0_rescreen_confirmation(
    pipeline_path: str | Path,
) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    rows: list[dict[str, Any]] = []
    seed42 = {
        str(job["experiment_id"]): job for job in wave0_rescreen_jobs(pipeline)
        if str(job["vision_encoder"]) == "dinov3_vits16"
    }
    for experiment_id, job in seed42.items():
        path = output / "wave0-rescreen" / str(job["run_id"]) / "metrics.json"
        rows.append(
            {
                "experiment_id": experiment_id,
                "text_encoder": str(job["text_encoder"]),
                "seed": 42,
                "dev_R@1": float(json.loads(path.read_text())["mean_R@1"]),
            }
        )
    for job in wave0_rescreen_confirmation_jobs(pipeline):
        path = (
            output / "wave0-rescreen-confirmation" / str(job["run_id"])
            / "metrics.json"
        )
        if not path.is_file():
            raise RuntimeError(f"missing top-three confirmation result {path}")
        rows.append(
            {
                "experiment_id": str(job["experiment_id"]),
                "text_encoder": str(job["text_encoder"]),
                "seed": int(job["seed"]),
                "dev_R@1": float(json.loads(path.read_text())["mean_R@1"]),
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["experiment_id", "text_encoder"], as_index=False)
        .agg(
            mean_3seed=("dev_R@1", "mean"),
            std_3seed=("dev_R@1", "std"),
            minimum=("dev_R@1", "min"),
            maximum=("dev_R@1", "max"),
        )
        .sort_values("mean_3seed", ascending=False)
        .reset_index(drop=True)
    )
    summary["rank"] = summary.index + 1
    destination = output / "wave0-rescreen-confirmation"
    atomic_csv(frame.sort_values(["experiment_id", "seed"]), destination / "results.csv")
    atomic_csv(summary, destination / "summary.csv")
    result = {
        "status": "CONFIRMATION_COMPLETE_PAIR_DECISION_RECORDED",
        "wave1_unblocked": False,
        "locked_student": "dinov3_vits16__all_minilm_l6_v2",
        "student_pair_changed": False,
        "student_pair_decision_permanent_for_study": True,
        "adoption_status": "DECIDED_KEEP_MINILM",
        "decision_rationale": [
            (
                "MiniLM anchors the queue study, 18-cell screen, dose-response "
                "curve, v4 teacher caches, and all 31 claim-bearing runs."
            ),
            (
                "The E5 gain carries no headline claim and changes neither the "
                "queue mechanism nor the pair-ranking finding."
            ),
            (
                "E5 demonstrates that the recipe correction generalises across "
                "text encoders, so the queue effect is not a MiniLM artifact."
            ),
        ],
        "ranking_replication": {
            "e5_mean_percent": 38.883,
            "e5_sd_pp": 0.012,
            "bge_mean_percent": 38.262,
            "bge_sd_pp": 0.094,
            "minilm_mean_percent": 37.324,
            "minilm_sd_pp": 0.282,
            "ranking_replicated": True,
            "interval_overlap_observed": False,
            "queue_selected_pair_corrected_rank": 3,
            "e5_minus_minilm_pp": 1.56,
        },
        "stability_observation": (
            "The lowest-performing pair, MiniLM, also has the largest seed "
            "standard deviation. It appears to be a less stable alignment "
            "target as well as a weaker one; this is an observation, not a "
            "mechanism claim."
        ),
        "deferred_best_available_configuration": {
            "pair": "dinov3_vits16__e5_small_v2",
            "purpose": "efficiency_frontier_plot_only",
            "timing": "end_of_project",
            "part_of_main_study": False,
            "run_now": False,
        },
        "observed_three_seed_ranking": summary["experiment_id"].tolist(),
        "results": summary.to_dict(orient="records"),
    }
    atomic_json(result, destination / "report.json")
    atomic_text(
        "# Wave 0 top-three pair confirmation\n\n"
        "DINOv3 ViT-S/16 + MiniLM remains permanently locked for the main "
        "study. E5 is not adopted. The decision preserves continuity with the "
        "MiniLM queue curve, screen, teacher caches, and all 31 claim-bearing "
        "runs; Wave 1 remains blocked.\n\n"
        "E5 38.883% (SD 0.012pp), BGE 38.262% (0.094pp), and MiniLM "
        "37.324% (0.282pp) replicate the ranking without observed overlap. "
        "The queue-selected MiniLM pair is last by 1.56pp. MiniLM is also the "
        "most variable pair; this is reported as an observation, not a "
        "mechanism claim.\n\n"
        "E5 demonstrates that the correction generalises beyond MiniLM. One E5 "
        "best-available-configuration point is deferred to the end-of-project "
        "efficiency frontier and is outside the main study.\n\n"
        + summary.to_markdown(index=False)
        + "\n",
        destination / "report.md",
    )
    return result


def status(pipeline_path: str | Path) -> pd.DataFrame:
    _, pipeline = load_pipeline(pipeline_path)
    output = output_root(pipeline)
    rows = []
    base = load_config(ROOT / str(pipeline["base_config"]))
    if _probe_mode(pipeline):
        checks = {
            "validation": output / "manifests/validation.json",
            "prefetch": output / "prefetch.json",
            "teacher_cache": ROOT / str(base["distillation"]["cache_path"]),
            "distillation_selection": output / "selection/distillation.json",
            "probe_report": output / "probe_report.json",
            "report": output / "report.json",
        }
    else:
        checks = {
            "validation": output / "manifests/validation.json",
            "prefetch": output / "prefetch.json",
            "pair_selection": output / "selection/pair.json",
            "teacher_cache": ROOT / str(base["distillation"]["cache_path"]),
            "recipe_selection": output / "selection/recipe.json",
            "oracle": output / "oracle.json",
            "report": output / "report.json",
        }
        if not _direct_recovery_recipe_study(pipeline):
            checks["distillation_selection"] = output / "selection/distillation.json"
            checks["component_smoke"] = output / "component_smoke.json"
        if bool(pipeline.get("batch_probe", {}).get("enabled", True)):
            checks["batch_probe"] = output / "batch_probe/common_batch.json"
        if pipeline.get("recovery"):
            checks["batch_shortlist"] = output / "selection/batch_shortlist.json"
    rows.extend(
        {
            "stage": key,
            "state": "COMPLETED" if path.is_file() else "INCOMPLETE",
            "complete": path.is_file(),
            "target": str(path),
        }
        for key, path in checks.items()
    )
    stage_jobs = (
        (("sensitivity", sensitivity_jobs(pipeline)),)
        if _probe_mode(pipeline)
        else (
            ("reference", [{"run_id": x} for x in pipeline["references"]]),
            ("pair", pair_jobs(pipeline)),
            ("sensitivity", sensitivity_jobs(pipeline)),
            ("ablation", ablation_jobs(pipeline)),
            ("final", final_jobs(pipeline)),
            ("transfer", transfer_jobs(pipeline)),
        )
    )
    for stage, jobs in stage_jobs:
        folder = "references" if stage == "reference" else stage
        for job in jobs:
            path = output / folder / str(job["run_id"]) / "metrics.json"
            rows.append(
                {
                    "stage": stage,
                    "run_id": job["run_id"],
                    "state": "FINGERPRINT_VALID_COMPLETED"
                    if path.is_file() and (path.parent / "fingerprint.json").is_file()
                    else "INCOMPLETE",
                    "complete": path.is_file(),
                    "target": str(path),
                }
            )
    if pipeline.get("recovery") and (output / "selection/batch_shortlist.json").is_file():
        for job in recovery_confirmation_jobs(pipeline):
            path = output / "confirmation" / str(job["run_id"]) / "metrics.json"
            rows.append(
                {
                    "stage": "confirmation",
                    "run_id": job["run_id"],
                    "state": "FINGERPRINT_VALID_COMPLETED"
                    if path.is_file() and (path.parent / "fingerprint.json").is_file()
                    else "INCOMPLETE",
                    "complete": path.is_file(),
                    "target": str(path),
                }
            )
    for marker in sorted((output / "slurm_status").glob("*/*/status.json")):
        payload = json.loads(marker.read_text())
        rows.append(
            {
                "stage": f"slurm:{payload.get('command', marker.parents[1].name)}",
                "run_id": marker.parent.name,
                "state": payload.get("status", "UNKNOWN"),
                "complete": payload.get("status") == "COMPLETED",
                "target": str(marker),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description="Alignment v3 experiment runner")
    parser.add_argument(
        "command",
        choices=(
            "validate", "prefetch", "reference", "batch-probe",
            "train-pair", "eval-pair", "select-pair",
            "select-recovery-batches", "train-confirmation", "eval-confirmation",
            "select-recovery-pair", "teacher-cache",
            "train-sensitivity", "eval-sensitivity", "select-distillation",
            "report-wave1", "report-wave1-teachers",
            "train-resolution", "eval-resolution",
            "oom-smoke-test", "summarize-oom-smoke",
            "train-wave0-lr-calibration", "eval-wave0-lr-calibration",
            "train-wave0-lr-control", "eval-wave0-lr-control",
            "eval-wave0-accepted", "train-wave0-winner-confirmation",
            "eval-wave0-winner-confirmation", "lock-wave0-corrected-recipe",
            "train-wave0-rescreen", "eval-wave0-rescreen",
            "train-wave0-rescreen-confirmation",
            "eval-wave0-rescreen-confirmation",
            "train-wave0-followup", "eval-wave0-followup",
            "train-wave0-screen", "eval-wave0-screen",
            "train-wave0-select", "eval-wave0-select",
            "select-wave0-lrsweep", "select-wave0-finalists",
            "select-wave0-recipe", "ledger-pending", "ledger-wave0-screen",
            "report-wave0-queue-ablation", "report-wave0-rescreen",
            "report-wave0-rescreen-confirmation", "ledger-pending-stage",
            "report-wave0-2080ti-pilot",
            "component-smoke",
            "train-ablation", "eval-ablation", "select-recipe",
            "select-recovery-recipe",
            "train-final", "eval-final", "oracle", "report", "status",
            "transfer",
        ),
    )
    parser.add_argument("--pipeline", default="configs/alignment_v3/pipeline.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--ledger-stage")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    command = args.command
    if command == "validate":
        result: Any = validate(args.pipeline)
    elif command == "prefetch":
        result = prefetch(args.pipeline)
    elif command == "reference":
        result = evaluate_reference(args.pipeline, args.index)
    elif command == "batch-probe":
        result = batch_probe(args.pipeline)
    elif command == "oom-smoke-test":
        result = wave0_oom_smoke_test(args.pipeline, args.index)
    elif command == "summarize-oom-smoke":
        result = summarize_wave0_oom(args.pipeline)
    elif command == "train-wave0-lr-calibration":
        result = run_wave0_lr_calibration(
            args.pipeline, args.index, evaluate=False, resume=not args.no_resume
        )
    elif command == "eval-wave0-lr-calibration":
        result = run_wave0_lr_calibration(
            args.pipeline, args.index, evaluate=True
        )
    elif command == "train-wave0-followup":
        result = run_wave0_followup(
            args.pipeline, args.index, evaluate=False, resume=not args.no_resume
        )
    elif command == "eval-wave0-followup":
        result = run_wave0_followup(
            args.pipeline, args.index, evaluate=True
        )
    elif command == "eval-wave0-accepted":
        result = evaluate_accepted_wave0_followup(args.pipeline, args.index)
    elif command == "lock-wave0-corrected-recipe":
        result = lock_wave0_corrected_recipe(args.pipeline)
    elif command == "select-wave0-lrsweep":
        result = select_wave0_lr_sweep(args.pipeline)
    elif command == "select-wave0-finalists":
        result = select_wave0_finalists(args.pipeline)
    elif command == "select-wave0-recipe":
        result = select_wave0_recipe(args.pipeline)
    elif command == "report-wave0-queue-ablation":
        result = report_wave0_queue_ablation(args.pipeline)
    elif command == "report-wave0-2080ti-pilot":
        result = report_wave0_2080ti_pilot(args.pipeline)
    elif command == "report-wave0-rescreen":
        result = report_wave0_rescreen(args.pipeline)
    elif command == "report-wave0-rescreen-confirmation":
        result = report_wave0_rescreen_confirmation(args.pipeline)
    elif command == "ledger-pending":
        if not args.ledger_stage:
            parser.error("--ledger-stage is required for ledger-pending")
        result = ledger_pending(args.pipeline, args.ledger_stage, args.index)
    elif command == "ledger-pending-stage":
        if not args.ledger_stage:
            parser.error("--ledger-stage is required for ledger-pending-stage")
        result = ledger_pending_stage(args.pipeline, args.ledger_stage)
    elif command == "ledger-wave0-screen":
        result = ledger_pending_stage(args.pipeline, "wave0-screen")
    elif command.startswith("train-"):
        result = run_training_stage(args.pipeline, command.removeprefix("train-"), args.index, resume=not args.no_resume)
    elif command.startswith("eval-"):
        result = evaluate_training_stage(args.pipeline, command.removeprefix("eval-"), args.index)
    elif command == "select-pair":
        result = select_pair(args.pipeline)
    elif command == "select-recovery-batches":
        result = select_recovery_batches(args.pipeline)
    elif command == "select-recovery-pair":
        result = select_recovery_pair(args.pipeline)
    elif command == "teacher-cache":
        result = build_teacher_cache(args.pipeline)
    elif command == "select-distillation":
        result = select_distillation(args.pipeline)
    elif command == "report-wave1":
        result = report_wave1(args.pipeline)
    elif command == "report-wave1-teachers":
        result = report_wave1_teacher_comparison()
    elif command == "component-smoke":
        result = component_smoke(args.pipeline)
    elif command == "select-recipe":
        result = select_recipe(args.pipeline)
    elif command == "select-recovery-recipe":
        result = select_recovery_recipe(args.pipeline)
    elif command == "oracle":
        result = oracle(args.pipeline)
    elif command == "transfer":
        result = evaluate_transfer(args.pipeline, args.index)
    elif command == "report":
        result = report(args.pipeline)
    else:
        result = status(args.pipeline)
    print(result.to_string(index=False) if isinstance(result, pd.DataFrame) else result)
    # Every stage exits 0 on a clean run, including stages that no-op because a
    # continuation gate upstream did not pass (status SKIPPED_GATE /
    # STOPPED_AT_GATE) -- that is intentional so the job graph still reaches
    # `report` and produces a report.json even when the science stopped early.
    # But it means `sacct` shows the whole graph as COMPLETED regardless of
    # whether the experiment actually finished, which reads as false success.
    # The terminal `report` job is the one place we can surface that honestly
    # without disturbing any `afterok` dependency in the submission graph:
    # exit non-zero there whenever the pipeline did not reach real completion.
    # Exit code 3 (as opposed to Python's default 1 on an unhandled
    # exception) marks "ran cleanly but stopped short", distinguishable from
    # a genuine crash.
    if command == "report" and isinstance(result, dict) and result.get("status") != "COMPLETE":
        sys.exit(3)
    if (
        command == "select-wave0-finalists"
        and isinstance(result, dict)
        and not result.get("proceed", False)
    ):
        sys.exit(3)


if __name__ == "__main__":
    main()
