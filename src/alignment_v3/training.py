from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.alignment_v3.distillation import TeacherCache, distillation_loss
from src.alignment_v3.fingerprint import Fingerprint, is_fresh, write_fingerprint
from src.alignment_v3.model import AlignmentV3Model, build_model
from src.alignment_v3.queue_diagnostics import ProjectorDriftTracker, append_jsonl
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_json
from src.training.evaluate import evaluate_model
from src.training.losses import (
    CrossBatchMemory,
    contrastive_loss_with_memory,
    sigmoid_contrastive_loss,
)
from src.utils.config import save_config
from src.utils.device import get_device
from src.utils.logging import CSVLogger
from src.utils.seed import set_seed


def _load_teacher_caches(config: dict[str, Any]) -> dict[str, TeacherCache]:
    if not bool(config.get("recipe", {}).get("distillation", False)):
        return {}
    distillation = config["distillation"]
    configured = distillation.get("teachers")
    if configured is None:
        return {"default": TeacherCache(str(distillation["cache_path"]))}
    teachers: dict[str, TeacherCache] = {}
    for item in configured:
        name = str(item["name"])
        if name in teachers:
            raise ValueError(f"duplicate distillation teacher name {name!r}")
        teachers[name] = TeacherCache(str(item["cache_path"]))
    if not teachers:
        raise ValueError("multi-teacher distillation requires at least one teacher")
    weights = [float(item["weight"]) for item in configured]
    if any(weight < 0 for weight in weights):
        raise ValueError("teacher loss weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ValueError(f"teacher loss weights must sum to 1.0, got {weights}")
    return teachers


def _distillation_auxiliary(
    config: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    teachers: dict[str, TeacherCache],
    image_ids: list[str],
    text_ids: list[str],
    captions: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not teachers:
        return torch.zeros((), device=device), {}
    distillation = config["distillation"]
    configured = distillation.get("teachers")
    if configured is None:
        items = [
            {
                "name": "default",
                "weight": 1.0,
                "temperature": distillation.get("temperature", 2.0),
                "image_cosine_weight": distillation.get("image_cosine_weight", 0.25),
                "text_cosine_weight": distillation.get("text_cosine_weight", 0.25),
                "kl_weight": distillation.get("kl_weight", 1.0),
            }
        ]
    else:
        items = list(configured)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, float] = {}
    for item in items:
        name = str(item["name"])
        teacher = teachers[name]
        teacher_image, teacher_text = teacher.lookup(
            image_ids, text_ids, captions, device
        )
        if configured is None:
            teacher_outputs = outputs
        else:
            spaces = outputs.get("teacher_spaces")
            if not isinstance(spaces, dict) or name not in spaces:
                raise KeyError(f"model output is missing teacher space {name!r}")
            teacher_outputs = {
                "teacher_space_image": spaces[name]["image"],
                "teacher_space_text": spaces[name]["text"],
                "logit_scale": outputs["logit_scale"],
            }
        loss, components = distillation_loss(
            teacher_outputs,
            teacher_image,
            teacher_text,
            teacher_scale=float(teacher.metadata.get("logit_scale", 1.0)),
            temperature=float(item.get("temperature", distillation.get("temperature", 2.0))),
            image_cosine_weight=float(
                item.get(
                    "image_cosine_weight",
                    distillation.get("image_cosine_weight", 0.25),
                )
            ),
            text_cosine_weight=float(
                item.get(
                    "text_cosine_weight",
                    distillation.get("text_cosine_weight", 0.25),
                )
            ),
            kl_weight=float(item.get("kl_weight", distillation.get("kl_weight", 1.0))),
        )
        weight = float(item["weight"])
        total = total + weight * loss
        diagnostics[f"{name}_weighted_loss"] = weight * float(loss.detach())
        for component, value in components.items():
            diagnostics[f"{name}_{component}"] = float(value.detach())
    return total, diagnostics


def _optimizer_parameters(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "logit_scale" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _assert_parameter_contract(
    model: AlignmentV3Model,
    config: dict[str, Any],
    summary: dict[str, int],
) -> None:
    contract = config.get("provenance", {}).get("parameter_contract")
    if not contract:
        return
    for key, expected in contract.items():
        if key not in summary:
            raise KeyError(f"unknown parameter-contract field {key!r}")
        if int(summary[key]) != int(expected):
            raise RuntimeError(
                f"parameter contract failed for {key}: "
                f"loaded={summary[key]}, expected={expected}"
            )
    leaked = [
        key
        for key in model.inference_state_dict()
        if key.startswith(("teacher_image_head.", "teacher_text_head.", "teacher_heads."))
    ]
    if leaked:
        raise RuntimeError(f"training-only teacher heads leaked into inference state: {leaked[:3]}")


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(total_steps * warmup_fraction)

    def value(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, value)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with ``map_location=device`` so model and
    # optimizer tensors land directly on the training GPU. That also moves RNG
    # byte tensors to CUDA, but both RNG restoration APIs require CPU
    # ByteTensors. Normalising here preserves exact resume and also supports
    # legacy list/array representations.
    def cpu_byte_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [cpu_byte_tensor(value) for value in state["cuda"]]
        )


def _memory_state(memory: CrossBatchMemory) -> dict[str, Any]:
    return {
        "capacity": memory.capacity,
        "enable_image": memory.enable_image,
        "enable_text": memory.enable_text,
        "image_embeddings": memory.image_embeddings,
        "text_embeddings": memory.text_embeddings,
        "image_features": memory.image_features,
        "text_features": memory.text_features,
        "store_features": memory.store_features,
        "image_semantic_embeddings": memory.image_semantic_embeddings,
        "text_semantic_embeddings": memory.text_semantic_embeddings,
        "store_semantic_embeddings": memory.store_semantic_embeddings,
        "image_ids": memory.image_ids,
        "text_image_ids": memory.text_image_ids,
        "image_insertion_steps": memory.image_insertion_steps,
        "text_insertion_steps": memory.text_insertion_steps,
        "image_exposures": memory.image_exposures,
        "text_exposures": memory.text_exposures,
    }


def _restore_memory(memory: CrossBatchMemory, state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    memory.store_features = bool(state.get("store_features", memory.store_features))
    memory.store_semantic_embeddings = bool(
        state.get("store_semantic_embeddings", memory.store_semantic_embeddings)
    )
    for name in (
        "image_features",
        "text_features",
        "image_semantic_embeddings",
        "text_semantic_embeddings",
    ):
        value = state.get(name)
        setattr(memory, name, value.to(device) if value is not None else None)
    memory.image_embeddings = (
        state["image_embeddings"].to(device) if state.get("image_embeddings") is not None else None
    )
    memory.text_embeddings = (
        state["text_embeddings"].to(device) if state.get("text_embeddings") is not None else None
    )
    memory.image_ids = list(state.get("image_ids", []))
    memory.text_image_ids = list(state.get("text_image_ids", []))
    memory.image_insertion_steps = list(state.get("image_insertion_steps", []))
    memory.text_insertion_steps = list(state.get("text_insertion_steps", []))
    memory.image_exposures = list(state.get("image_exposures", []))
    memory.text_exposures = list(state.get("text_exposures", []))


def _save_checkpoint(
    path: Path,
    *,
    model: AlignmentV3Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    memory: CrossBatchMemory,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    fingerprint: Fingerprint,
    global_step: int = 0,
    diagnostic_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "memory_state": _memory_state(memory),
            "rng_state": _rng_state(),
            "epoch": epoch,
            "global_step": int(global_step),
            "diagnostic_state": diagnostic_state,
            "config": config,
            "metrics": metrics,
            "fingerprint": fingerprint.to_dict(),
            "parameter_summary": model.parameter_summary(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _link_epoch_snapshot(latest: Path, epoch: int) -> Path:
    """Freeze the just-written latest checkpoint without another serialization."""
    destination = latest.parent / f"epoch_{int(epoch):02d}.pt"
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        if int(saved.get("epoch", -1)) != int(epoch):
            raise RuntimeError(
                f"immutable epoch snapshot mismatch: {destination} "
                f"contains epoch {saved.get('epoch')}"
            )
        return destination
    os.link(latest, destination)
    return destination


def _save_inference_export(
    path: Path,
    model: AlignmentV3Model,
    config: dict[str, Any],
    fingerprint: Fingerprint,
    metrics: dict[str, float],
) -> None:
    inference_config = deepcopy(config)
    # Distillation heads only exist to match the teacher during training.
    # The deployment model reconstructs without them and loads strictly.
    inference_config["recipe"]["distillation"] = False
    training_summary = model.parameter_summary()
    inference_summary = {
        "params_total_inference": int(training_summary["params_total_inference"]),
        "params_trainable_inference": int(
            training_summary["params_trainable_inference"]
        ),
        "params_training_only_excluded": int(
            training_summary["params_training_only"]
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.inference_state_dict(),
            "config": inference_config,
            "training_recipe": deepcopy(config["recipe"]),
            "fingerprint": fingerprint.to_dict(),
            "metrics": metrics,
            "parameter_summary": inference_summary,
            "training_parameter_summary": training_summary,
            "training_only_heads_removed": True,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_training_checkpoint(
    path: str | Path,
    model: AlignmentV3Model,
    *,
    device: torch.device,
    expected_fingerprint: Fingerprint | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if expected_fingerprint is not None:
        actual = Fingerprint.from_dict(checkpoint["fingerprint"])
        if actual.digest != expected_fingerprint.digest:
            raise ValueError(f"checkpoint fingerprint mismatch for {path}")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def load_inference_export(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[AlignmentV3Model, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _precision(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype, bool]:
    requested = str(config["training"].get("precision", "bf16")).lower()
    if device.type != "cuda":
        return False, torch.float32, False
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but the selected GPU does not support BF16")
        return True, torch.bfloat16, False
    if requested == "fp16":
        return True, torch.float16, True
    if requested in {"fp32", "float32"}:
        return False, torch.float32, False
    raise ValueError(f"unsupported precision {requested!r}")


def train(
    config: dict[str, Any],
    fingerprint: Fingerprint,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=bool(config.get("deterministic", False)))
    device = get_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(config["training"].get("allow_tf32", True))
        torch.cuda.reset_peak_memory_stats(device)
    train_loader, dev_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    if config["model"]["vision_encoder"] == "dinov3_convnext_tiny" and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    parameters = model.parameter_summary()
    _assert_parameter_contract(model, config, parameters)
    budgets = config.get("budgets", {})
    if parameters["params_total_inference"] > int(budgets.get("max_total_params", 80_000_000)):
        raise ValueError(f"inference parameter budget exceeded: {parameters}")
    if parameters["params_trainable_inference"] > int(budgets.get("max_trainable_params", 5_000_000)):
        raise ValueError(f"inference-trainable parameter budget exceeded: {parameters}")

    train_cfg = config["training"]
    save_dir = Path(str(train_cfg["save_dir"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")
    write_fingerprint(save_dir / "fingerprint.json", fingerprint)
    optimizer_kwargs = {
        "lr": float(train_cfg.get("lr", 3e-4)),
        "betas": tuple(float(value) for value in train_cfg.get("betas", [0.9, 0.98])),
    }
    try:
        optimizer = torch.optim.AdamW(
            _optimizer_parameters(model, float(train_cfg.get("weight_decay", 0.01))),
            fused=device.type == "cuda",
            **optimizer_kwargs,
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            _optimizer_parameters(model, float(train_cfg.get("weight_decay", 0.01))),
            **optimizer_kwargs,
        )
    epochs = int(train_cfg.get("epochs", 12))
    schedule_steps = int(
        train_cfg.get("max_optimizer_steps") or max(1, epochs * len(train_loader))
    )
    scheduler = _scheduler(
        optimizer, max(1, schedule_steps), float(train_cfg.get("warmup_fraction", 0.05))
    )
    amp_enabled, amp_dtype, scale_gradients = _precision(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=scale_gradients)
    queue_mode = str(train_cfg.get("queue_mode", "both"))
    if queue_mode not in {"none", "image_only", "text_only", "both", "both_fresh"}:
        raise ValueError(f"unsupported queue_mode {queue_mode!r}")
    # both_fresh is the identification control: an identical negative set to
    # "both", re-projected through the current projector so that the entries
    # carry no staleness.  Any difference in outcome between both and
    # both_fresh is attributable to staleness alone.
    fresh_queue = queue_mode == "both_fresh"
    queue_filter_mode = str(train_cfg.get("queue_filter_mode", "none"))
    if queue_filter_mode not in {"none", "semantic", "matched_random"}:
        raise ValueError(f"unsupported queue_filter_mode {queue_filter_mode!r}")
    queue_weight_mode = str(train_cfg.get("queue_weight_mode", "none"))
    if queue_weight_mode not in {"none", "match_inbatch"}:
        raise ValueError(f"unsupported queue_weight_mode {queue_weight_mode!r}")
    semantic_cache = None
    if queue_filter_mode != "none":
        cache_path = train_cfg.get("queue_semantic_cache")
        if not cache_path:
            raise ValueError("queue filtering requires training.queue_semantic_cache")
        resolved_cache = Path(str(cache_path))
        if not resolved_cache.is_absolute():
            resolved_cache = Path(__file__).resolve().parents[2] / resolved_cache
        semantic_cache = TeacherCache(resolved_cache)
    memory = CrossBatchMemory(
        int(train_cfg.get("memory_queue_size", 16384)),
        enable_image=queue_mode in {"image_only", "both", "both_fresh"},
        enable_text=queue_mode in {"text_only", "both", "both_fresh"},
        store_features=fresh_queue,
        store_semantic_embeddings=semantic_cache is not None,
    )
    if fresh_queue and (model.token_aggregator is not None or model.adapter is not None):
        raise ValueError(
            "both_fresh removes projector staleness only; this configuration has a "
            "trainable aggregator/adapter upstream of the projector, whose staleness "
            "would remain and confound the control"
        )
    # ResidualProjectionHead carries Dropout(0.1).  Re-projecting in train mode
    # would resample it, so a fresh resident would differ from its stale
    # counterpart by dropout as well as by projector state.  Re-projecting in
    # eval mode makes the fresh entry the deterministic image of its frozen
    # feature under the current projector, leaving projector state as the only
    # systematic difference.  The residual nuisance -- that stale entries still
    # carry a training-time dropout sample the fresh ones do not -- is bounded
    # by the residual branch scale and is recorded in the run manifest.
    fresh_eval_mode = bool(train_cfg.get("fresh_projection_eval_mode", True))

    def _make_reprojector(module: torch.nn.Module):
        def reproject(features: torch.Tensor) -> torch.Tensor:
            if not fresh_eval_mode:
                return module(features)
            was_training = module.training
            module.eval()
            try:
                return module(features)
            finally:
                module.train(was_training)

        return reproject

    image_reprojector = _make_reprojector(model.image_projection) if fresh_queue else None
    text_reprojector = _make_reprojector(model.text_projection) if fresh_queue else None
    loss_type = str(config.get("recipe", {}).get("loss_type", "infonce_queue"))
    if loss_type not in {"infonce_queue", "infonce_no_queue", "sigmoid"}:
        raise ValueError(f"unsupported loss_type {loss_type!r}")
    teachers = _load_teacher_caches(config)
    diagnostics_cfg = config.get("queue_diagnostics", {})
    drift_tracker = None
    if bool(diagnostics_cfg.get("enabled", False)):
        drift_tracker = ProjectorDriftTracker(
            model,
            dev_loader,
            manifest_path=diagnostics_cfg["probe_manifest"],
            seed=int(diagnostics_cfg.get("probe_seed", 20260727)),
            size=int(diagnostics_cfg.get("probe_images", 512)),
            interval=int(diagnostics_cfg.get("drift_interval_steps", 10)),
        )
    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    latest = save_dir / "latest.pt"
    if resume and latest.is_file():
        checkpoint = load_training_checkpoint(
            latest, model, device=device, expected_fingerprint=fingerprint
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        _restore_memory(memory, checkpoint.get("memory_state"), device)
        if drift_tracker is not None:
            drift_tracker.load_state_dict(checkpoint.get("diagnostic_state") or {})
        _restore_rng(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"]) + 1
        if bool(train_cfg.get("save_every_epoch", False)):
            _link_epoch_snapshot(latest, int(checkpoint["epoch"]))
        best_path = save_dir / "best.pt"
        if best_path.is_file():
            saved_best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_metrics = dict(saved_best.get("metrics", {}))
            best_score = float(best_metrics.get("mean_R@1", -1))
            best_epoch = int(saved_best.get("epoch", 0))
    logger = CSVLogger(save_dir / "metrics.csv")
    fixed_schedule = not bool(train_cfg.get("select_on_dev", True))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    without_improvement = 0
    completed_epoch = start_epoch - 1
    global_step = 0
    if resume and latest.is_file():
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * len(train_loader)))
    started_run = time.perf_counter()
    diagnostics_path = save_dir / "queue_diagnostics.jsonl"
    max_optimizer_steps = train_cfg.get("max_optimizer_steps")
    reached_step_limit = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running: dict[str, float] = {
            "loss": 0.0,
            "contrastive": 0.0,
            "distillation": 0.0,
            "text_rows": 0.0,
            "unique_images": 0.0,
            "batches": 0.0,
        }
        examples = 0
        started_epoch = time.perf_counter()
        progress = tqdm(train_loader, desc=f"v3 epoch {epoch}", leave=False)
        for optimizer_step_in_epoch, batch in enumerate(progress, start=1):
            step_timing = bool(train_cfg.get("step_timing", False))
            if step_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            started_step = time.perf_counter()
            images = batch["images"].to(device, non_blocking=True)
            if config["model"]["vision_encoder"] == "dinov3_convnext_tiny" and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            captions = [str(value) for value in batch["captions"]]
            image_ids = [str(value) for value in batch["image_paths"]]
            text_ids = [str(value) for value in batch.get("text_image_paths", image_ids)]
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                outputs = model(images, captions)
                semantic_image = semantic_text = None
                if semantic_cache is not None:
                    semantic_image, semantic_text = semantic_cache.lookup(
                        image_ids, text_ids, captions, device
                    )
                if loss_type == "sigmoid":
                    contrastive = sigmoid_contrastive_loss(
                        outputs["image_embeds"],
                        outputs["text_embeds"],
                        outputs["sigmoid_logit_scale"],
                        outputs["sigmoid_logit_bias"],
                        image_ids,
                        text_ids,
                    )
                else:
                    loss_diagnostics = contrastive_loss_with_memory(
                        outputs["image_embeds"],
                        outputs["text_embeds"],
                        outputs["logit_scale"],
                        image_ids,
                        text_ids,
                        memory,
                        return_diagnostics=True,
                        image_reprojector=image_reprojector,
                        text_reprojector=text_reprojector,
                        semantic_image_embeddings=semantic_image,
                        semantic_text_embeddings=semantic_text,
                        queue_filter_mode=queue_filter_mode,
                        queue_filter_threshold=float(
                            train_cfg.get("queue_filter_threshold", 0.222935)
                        ),
                        queue_filter_seed=int(config["seed"]) * 1_000_003 + global_step,
                        queue_weight_mode=queue_weight_mode,
                    )
                    contrastive = loss_diagnostics.total
                auxiliary = torch.zeros((), device=device)
                if teachers:
                    auxiliary, _ = _distillation_auxiliary(
                        config,
                        outputs,
                        teachers,
                        image_ids,
                        text_ids,
                        captions,
                        device,
                    )
                loss = float(train_cfg.get("contrastive_weight", 1.0)) * contrastive + float(
                    config.get("distillation", {}).get("strength", 1.0)
                ) * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip = float(train_cfg.get("gradient_clip_norm", 1.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            queue_enabled = loss_type == "infonce_queue" or (
                loss_type == "infonce_no_queue" and queue_mode != "none"
            )
            if queue_enabled:
                # The resident queue participated in the loss just computed,
                # so count the current exposure before recording diagnostics.
                memory.record_exposure()
                queue_state = memory.diagnostics(global_step)
                memory.enqueue(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    image_ids,
                    text_ids,
                    optimizer_step=global_step,
                    image_features=outputs["image_pre_projection"] if fresh_queue else None,
                    text_features=outputs["text_pre_projection"] if fresh_queue else None,
                    image_semantic_embeddings=semantic_image,
                    text_semantic_embeddings=semantic_text,
                )
                diagnostic_row = {
                    "optimizer_step": global_step,
                    "epoch": epoch,
                    "optimizer_step_in_epoch": optimizer_step_in_epoch,
                    "phase": "full_queue" if queue_state["full"] else "warmup",
                    **queue_state,
                    "i2t_loss": float(loss_diagnostics.i2t.detach()),
                    "t2i_loss": float(loss_diagnostics.t2i.detach()),
                    "image_queue_positive_fraction": (
                        loss_diagnostics.image_queue_positive_fraction
                    ),
                    "text_queue_positive_fraction": (
                        loss_diagnostics.text_queue_positive_fraction
                    ),
                    "image_queue_filtered_fraction": (
                        loss_diagnostics.image_queue_filtered_fraction
                    ),
                    "text_queue_filtered_fraction": (
                        loss_diagnostics.text_queue_filtered_fraction
                    ),
                    "queue_filter_mode": queue_filter_mode,
                    "queue_weight_mode": queue_weight_mode,
                }
                if step_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - started_step
                diagnostic_row["step_seconds"] = step_seconds
                diagnostic_row["images_per_second"] = len(image_ids) / max(
                    step_seconds, 1e-9
                )
                append_jsonl(diagnostics_path, diagnostic_row)
            if drift_tracker is not None:
                drift_row = drift_tracker.measure(global_step)
                if drift_row is not None:
                    drift_row["epoch"] = epoch
                    drift_row["optimizer_step_in_epoch"] = optimizer_step_in_epoch
                    append_jsonl(save_dir / "projector_drift.jsonl", drift_row)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))
            batch_images = len(image_ids)
            examples += batch_images
            running["loss"] += float(loss.detach())
            running["contrastive"] += float(contrastive.detach())
            running["distillation"] += float(auxiliary.detach())
            running["text_rows"] += len(text_ids)
            running["unique_images"] += len(image_ids)
            running["batches"] += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if max_optimizer_steps is not None and global_step >= int(max_optimizer_steps):
                reached_step_limit = True
                break
        elapsed = time.perf_counter() - started_epoch
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": running["loss"] / max(1, len(train_loader)),
            "contrastive_loss": running["contrastive"] / max(1, len(train_loader)),
            "distillation_loss": running["distillation"] / max(1, len(train_loader)),
            "mean_loss_magnitude": running["contrastive"] / max(1, len(train_loader)),
            "unique_images_per_batch": running["unique_images"] / max(1, running["batches"]),
            "text_rows_per_batch": running["text_rows"] / max(1, running["batches"]),
            "images_per_second": examples / max(elapsed, 1e-9),
            "epoch_seconds": elapsed,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "peak_training_memory_bytes": float(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "gpu_total_memory_bytes": float(
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else 0
            ),
        }
        metrics: dict[str, float] = {}
        if not fixed_schedule:
            metrics = evaluate_model(
                model,
                dev_loader,
                device,
                list(config.get("evaluation", {}).get("k_values", [1, 5, 10])),
            )
            row.update(metrics)
        logger.write(row)
        print(row)
        checkpoint_metrics = {**metrics, **parameters, **{k: float(v) for k, v in row.items()}}
        _save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            memory=memory,
            epoch=epoch,
            config=config,
            metrics=checkpoint_metrics,
            fingerprint=fingerprint,
            global_step=global_step,
            diagnostic_state=(
                drift_tracker.state_dict() if drift_tracker is not None else None
            ),
        )
        if bool(train_cfg.get("save_every_epoch", False)):
            _link_epoch_snapshot(latest, epoch)
        if fixed_schedule or float(metrics["mean_R@1"]) > best_score:
            best_score = float(metrics.get("mean_R@1", -1))
            best_epoch = epoch
            best_metrics = checkpoint_metrics
            without_improvement = 0
            _save_checkpoint(
                save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                memory=memory,
                epoch=epoch,
                config=config,
                metrics=checkpoint_metrics,
                fingerprint=fingerprint,
                global_step=global_step,
                diagnostic_state=(
                    drift_tracker.state_dict() if drift_tracker is not None else None
                ),
            )
        else:
            without_improvement += 1
        completed_epoch = epoch
        early_stopping_enabled = not bool(
            train_cfg.get("disable_early_stopping", False)
        )
        if reached_step_limit or (
            early_stopping_enabled
            and not fixed_schedule
            and without_improvement >= patience
        ):
            break

    best_path = save_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("training did not produce best.pt")
    best_checkpoint = load_training_checkpoint(
        best_path, model, device=device, expected_fingerprint=fingerprint
    )
    _save_inference_export(
        save_dir / "inference.pt", model, config, fingerprint, best_checkpoint["metrics"]
    )
    summary = {
        "status": "COMPLETE",
        "seed": seed,
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "fixed_schedule": fixed_schedule,
        "stopped_early": completed_epoch < epochs,
        "wall_seconds": time.perf_counter() - started_run,
        "best_metrics": best_metrics,
        "parameter_summary": parameters,
        "fingerprint_digest": fingerprint.digest,
    }
    atomic_json(summary, save_dir / "run_summary.json")
    return summary
