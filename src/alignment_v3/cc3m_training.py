from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from src.alignment_v3.efficiency_frontier import _hardware_guard
from src.alignment_v3.fingerprint import (
    Fingerprint,
    code_fingerprint,
    hash_config,
    sha256_file,
    write_fingerprint,
)
from src.alignment_v3.model import DINO_VISION_MODELS, build_model
from src.alignment_v3.training import (
    _optimizer_parameters,
    _precision,
    _restore_rng,
    _rng_state,
    _scheduler,
)
from src.data import build_dataloaders
from src.data.cc3m_tar import CursorSampler, IndexedTarDataset, worker_init_noop
from src.data.collate import image_text_collate
from src.models.text_encoders import TEXT_MODEL_REGISTRY
from src.phase15.io_utils import atomic_json
from src.training.evaluate import evaluate_model
from src.training.losses import contrastive_loss_with_memory
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = "configs/cc3m_scale/pipeline.yaml"
CODE_PATHS = [
    "src/alignment_v3/cc3m_training.py",
    "src/data/cc3m_tar.py",
    "src/alignment_v3/model.py",
    "src/alignment_v3/training.py",
    "src/training/losses.py",
]


def _pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return load_config(value if value.is_absolute() else ROOT / value)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _ccroot(pipeline: dict[str, Any]) -> Path:
    return _path(pipeline["root"])


def _acquisition(pipeline: dict[str, Any]) -> dict[str, Any]:
    path = _ccroot(pipeline) / "manifests/acquisition_complete.json"
    if not path.is_file():
        raise RuntimeError("CC3M acquisition is not complete")
    payload = json.loads(path.read_text())
    selected = _ccroot(pipeline) / "manifests/selected_keys.jsonl"
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("target_usable_pairs")
        != int(pipeline["acquisition"]["target_usable_pairs"])
        or payload.get("selected_key_manifest_sha256") != sha256_file(selected)
    ):
        raise RuntimeError("CC3M acquisition integrity gate failed")
    return payload


def build_tar_index(pipeline: dict[str, Any]) -> dict[str, Any]:
    acquisition = _acquisition(pipeline)
    selected_path = _ccroot(pipeline) / "manifests/selected_keys.jsonl"
    output = _ccroot(pipeline) / "manifests/selected_tar_index.parquet"
    manifest_path = _ccroot(pipeline) / "manifests/selected_tar_index.json"
    if output.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("selected_key_manifest_sha256")
            == acquisition["selected_key_manifest_sha256"]
            and existing.get("index_sha256") == sha256_file(output)
            and existing.get("rows")
            == int(pipeline["acquisition"]["target_usable_pairs"])
        ):
            return {**existing, "action": "REUSED_VERIFIED"}
        raise RuntimeError("tar index exists but does not match selected-key manifest")

    selected_by_shard: dict[str, dict[str, dict[str, Any]]] = {}
    with selected_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            selected_by_shard.setdefault(row["shard"], {})[row["key"]] = row
    offsets: dict[str, dict[str, tuple[int, int]]] = {}
    import tarfile

    for shard_name, selected in selected_by_shard.items():
        tar_path = _ccroot(pipeline) / "shards/train" / shard_name
        found: dict[str, tuple[int, int]] = {}
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if "." not in name:
                    continue
                key, extension = name.rsplit(".", 1)
                if (
                    key in selected
                    and extension.lower() in {"jpg", "jpeg", "png", "webp"}
                ):
                    if key in found:
                        raise RuntimeError(f"multiple image members for {shard_name}::{key}")
                    found[key] = (int(member.offset_data), int(member.size))
        missing = set(selected) - set(found)
        if missing:
            raise RuntimeError(
                f"{shard_name} missing {len(missing)} selected image members"
            )
        offsets[shard_name] = found

    columns: dict[str, list[Any]] = {
        "position": [],
        "canonical_id": [],
        "shard": [],
        "key": [],
        "tar_path": [],
        "offset": [],
        "size": [],
        "caption": [],
        "raw_sha256": [],
        "pixel_sha256": [],
    }
    with selected_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            offset, size = offsets[row["shard"]][row["key"]]
            values = {
                **row,
                "tar_path": str(
                    (_ccroot(pipeline) / "shards/train" / row["shard"]).resolve()
                ),
                "offset": offset,
                "size": size,
            }
            for key in columns:
                columns[key].append(values[key])
    table = pa.table(columns)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=65536)
    os.replace(temporary, output)
    result = {
        "status": "COMPLETE",
        "rows": table.num_rows,
        "selected_key_manifest_sha256": acquisition["selected_key_manifest_sha256"],
        "index_path": str(output.relative_to(ROOT)),
        "index_sha256": sha256_file(output),
        "format": "Parquet random-access tar member offsets",
        "images_extracted": False,
    }
    atomic_json(result, manifest_path)
    return result


def jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for arm in ("arm_a", "arm_b"):
        for seed in pipeline["training"]["seeds"]:
            result.append(
                {
                    "index": len(result),
                    "arm": arm,
                    "arm_id": pipeline["training"][arm]["id"],
                    "seed": int(seed),
                    "run_id": f"{pipeline['training'][arm]['id']}__seed_{int(seed)}",
                }
            )
    return result


def _job(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    selected = (
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if index is None
        else int(index)
    )
    values = jobs(pipeline)
    if selected < 0 or selected >= len(values):
        raise IndexError(selected)
    return values[selected]


def build_config(pipeline: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(load_config(_path(pipeline["base_config"])))
    training = pipeline["training"]
    arm = training[job["arm"]]
    config["seed"] = int(job["seed"])
    config["experiment_id"] = "cc3m_scale"
    config["run_id"] = job["run_id"]
    config["data"]["train_csv"] = str(
        (_ccroot(pipeline) / "manifests/selected_keys.jsonl").relative_to(ROOT)
    )
    config["data"]["val_csv"] = (
        str(arm["selection_dataset"])
        if job["arm"] == "arm_b"
        else "data/flickr30k/validation.csv"
    )
    config["data"]["group_by_image"] = True
    config["data"]["num_workers"] = int(training["dataloader_workers"])
    config["training"]["batch_size"] = int(training["batch_size"])
    config["training"]["drop_last"] = True
    config["training"]["lr"] = float(training["resolved_lr"])
    config["training"]["memory_queue_size"] = 0
    config["training"]["queue_mode"] = "none"
    config["training"]["precision"] = "bf16"
    config["training"]["max_optimizer_steps"] = int(arm["max_optimizer_steps"])
    config["training"]["checkpoint_interval_steps"] = int(
        training["checkpoint_interval_steps"]
    )
    config["training"]["passes"] = (
        1
        if job["arm"] == "arm_a"
        else int(arm["max_optimizer_steps"]) // 1320
    )
    config["training"]["select_on_dev"] = job["arm"] == "arm_b"
    config["training"]["early_stopping_patience"] = (
        int(arm["early_stopping_patience_evaluations"])
        if job["arm"] == "arm_b"
        else -1
    )
    config["training"]["save_dir"] = str(
        (_path(pipeline["checkpoint_root"]) / job["run_id"]).relative_to(ROOT)
    )
    config["recipe"]["loss_type"] = "infonce_no_queue"
    config["recipe"]["memory_queue_size"] = 0
    config["recipe"]["captions_per_image"] = 1
    config["provenance"]["cc3m_scale"] = {
        "arm": job["arm"],
        "arm_id": job["arm_id"],
        "repo_id": pipeline["acquisition"]["repo_id"],
        "revision": pipeline["acquisition"]["revision"],
        "selected_key_manifest_sha256": _acquisition(pipeline)[
            "selected_key_manifest_sha256"
        ],
        "tar_index_sha256": json.loads(
            (_ccroot(pipeline) / "manifests/selected_tar_index.json").read_text()
        )["index_sha256"],
        "flickr_test_evaluated": False,
        "lr_tuned_for_cc3m": False,
        "arm_b_selection_influenced": job["arm"] == "arm_b",
    }
    return config


def _fingerprint(config: dict[str, Any]) -> Fingerprint:
    provenance = config["provenance"]["cc3m_scale"]
    return Fingerprint(
        config_hash=hash_config(config),
        dataset_split_hash=provenance["selected_key_manifest_sha256"],
        model_checkpoint_id=(
            f"{DINO_VISION_MODELS[config['model']['vision_encoder']]}+"
            f"{TEXT_MODEL_REGISTRY[config['model']['text_encoder']]}"
        ),
        cache_version="none",
        preprocessing_hash=hash_config(
            {
                key: config["data"].get(key)
                for key in ("image_size", "interpolation", "image_mean", "image_std")
            }
        ),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
        seed=int(config["seed"]),
    )


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    acquisition = _acquisition(pipeline)
    index = build_tar_index(pipeline)
    if int(index["rows"]) != 1320 * int(pipeline["training"]["batch_size"]):
        raise RuntimeError("selected index does not produce exactly 1,320 full batches")
    configs = []
    for job in jobs(pipeline):
        config = build_config(pipeline, job)
        configs.append(
            {
                **job,
                "max_optimizer_steps": config["training"]["max_optimizer_steps"],
                "passes": config["training"]["passes"],
                "resolved_lr": config["training"]["lr"],
                "queue_mode": config["training"]["queue_mode"],
                "config_hash": hash_config(config),
            }
        )
    result = {
        "status": "READY",
        "acquisition_sha256": sha256_file(
            _ccroot(pipeline) / "manifests/acquisition_complete.json"
        ),
        "selected_key_manifest_sha256": acquisition["selected_key_manifest_sha256"],
        "tar_index_sha256": index["index_sha256"],
        "training_jobs": configs,
        "flickr_test_evaluation_jobs": 0,
    }
    atomic_json(result, _path(pipeline["output_root"]) / "manifests/validation.json")
    return result


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_NOT_A_GIT_CHECKOUT"


def _ledger_event(
    pipeline: dict[str, Any],
    job: dict[str, Any],
    fingerprint: Fingerprint,
    status: str,
    **values: Any,
) -> None:
    path = _path(pipeline["output_root"]) / "ledgers/training.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    question = (
        "Does unique CC3M-mirror data outperform repeated COCO data at "
        "matched optimizer-step compute?"
        if job["arm"] == "arm_a"
        else "What is the converged transfer value of the frozen-projector "
        "student on the selected CC3M-mirror subset?"
    )
    row = {
        "written_at_unix": time.time(),
        "status": status,
        "question": question,
        "run_id": job["run_id"],
        "arm": job["arm"],
        "seed": job["seed"],
        "config_fingerprint": fingerprint.digest,
        "git_sha": _git_sha(),
        **values,
    }
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_recovery(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    fingerprint: Fingerprint,
    global_step: int,
    pass_index: int,
    cursor: int,
    evaluation_complete: bool,
    best_score: float,
    best_pass: int,
    without_improvement: int,
) -> None:
    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _rng_state(),
            "config": config,
            "fingerprint": fingerprint.to_dict(),
            "global_step": global_step,
            "pass_index": pass_index,
            "cursor": cursor,
            "evaluation_complete": evaluation_complete,
            "best_score": best_score,
            "best_pass": best_pass,
            "without_improvement": without_improvement,
        },
        path,
    )


def _load_recovery(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    fingerprint: Fingerprint,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    actual = Fingerprint.from_dict(payload["fingerprint"])
    if actual.digest != fingerprint.digest:
        raise RuntimeError("refusing cross-config CC3M resume")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    _restore_rng(payload["rng_state"])
    return payload


def _make_train_loader(
    pipeline: dict[str, Any],
    config: dict[str, Any],
    *,
    pass_index: int,
    cursor: int,
) -> DataLoader:
    dataset = IndexedTarDataset(
        _ccroot(pipeline) / "manifests/selected_tar_index.parquet",
        run_seed=int(config["seed"]),
        pass_index=pass_index,
        image_size=int(config["data"]["image_size"]),
        image_mean=config["data"].get("image_mean"),
        image_std=config["data"].get("image_std"),
        interpolation=str(config["data"].get("interpolation", "bicubic")),
    )
    sampler = CursorSampler(len(dataset), cursor)
    workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=image_text_collate,
        worker_init_fn=worker_init_noop,
    )


_SIGNAL_REQUESTED = False


def _signal_handler(_: int, __: Any) -> None:
    global _SIGNAL_REQUESTED
    _SIGNAL_REQUESTED = True


def train_job(
    pipeline: dict[str, Any],
    index: int | None,
    *,
    stop_after_global_step: int | None = None,
) -> dict[str, Any]:
    global _SIGNAL_REQUESTED
    _SIGNAL_REQUESTED = False
    provenance = _hardware_guard(pipeline)
    job = _job(pipeline, index)
    config = build_config(pipeline, job)
    fingerprint = _fingerprint(config)
    save_dir = _path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")
    write_fingerprint(save_dir / "fingerprint.json", fingerprint)
    if (save_dir / "run_summary.json").is_file():
        summary = json.loads((save_dir / "run_summary.json").read_text())
        if summary.get("fingerprint") == fingerprint.digest:
            return {"status": "SKIPPED_FRESH", **job}
        raise RuntimeError("completed run has a different fingerprint")
    _ledger_event(
        pipeline,
        job,
        fingerprint,
        "STARTED_OR_RESUMED",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    set_seed(int(config["seed"]), deterministic=False)
    device = provenance.pop("device")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, float(config["training"]["weight_decay"])),
        lr=float(config["training"]["lr"]),
        betas=tuple(config["training"]["betas"]),
        fused=True,
    )
    total_steps = int(config["training"]["max_optimizer_steps"])
    scheduler = _scheduler(
        optimizer, total_steps, float(config["training"]["warmup_fraction"])
    )
    amp_enabled, amp_dtype, scale_gradients = _precision(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=scale_gradients)
    latest = save_dir / "latest.pt"
    global_step = 0
    pass_index = 0
    cursor = 0
    evaluation_complete = False
    best_score = -1.0
    best_pass = -1
    without_improvement = 0
    if latest.is_file():
        state = _load_recovery(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            fingerprint=fingerprint,
            device=device,
        )
        global_step = int(state["global_step"])
        pass_index = int(state["pass_index"])
        cursor = int(state["cursor"])
        evaluation_complete = bool(state["evaluation_complete"])
        best_score = float(state["best_score"])
        best_pass = int(state["best_pass"])
        without_improvement = int(state["without_improvement"])

    signal.signal(signal.SIGUSR1, _signal_handler)
    metrics_path = save_dir / "metrics.jsonl"
    pass_size = int(pipeline["acquisition"]["target_usable_pairs"])
    max_passes = int(config["training"]["passes"])
    started = time.perf_counter()
    while pass_index < max_passes and global_step < total_steps:
        if cursor < pass_size:
            loader = _make_train_loader(
                pipeline, config, pass_index=pass_index, cursor=cursor
            )
            model.train()
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                captions = [str(value) for value in batch["captions"]]
                image_ids = [str(value) for value in batch["image_paths"]]
                text_ids = [str(value) for value in batch["text_image_paths"]]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=amp_enabled, dtype=amp_dtype
                ):
                    outputs = model(images, captions)
                    loss = contrastive_loss_with_memory(
                        outputs["image_embeds"],
                        outputs["text_embeds"],
                        outputs["logit_scale"],
                        image_ids,
                        text_ids,
                        memory=None,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                with torch.no_grad():
                    model.logit_scale.clamp_(max=math.log(100.0))
                global_step += 1
                cursor += len(image_ids)
                if global_step % 10 == 0:
                    _append = {
                        "kind": "train_step",
                        "global_step": global_step,
                        "pass_index": pass_index,
                        "cursor": cursor,
                        "loss": float(loss.detach()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(_append, sort_keys=True) + "\n")
                checkpoint_due = (
                    global_step % int(config["training"]["checkpoint_interval_steps"])
                    == 0
                )
                requested_stop = (
                    stop_after_global_step is not None
                    and global_step >= stop_after_global_step
                )
                if checkpoint_due or _SIGNAL_REQUESTED or requested_stop:
                    _save_recovery(
                        latest,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        fingerprint=fingerprint,
                        global_step=global_step,
                        pass_index=pass_index,
                        cursor=cursor,
                        evaluation_complete=False,
                        best_score=best_score,
                        best_pass=best_pass,
                        without_improvement=without_improvement,
                    )
                if requested_stop:
                    _ledger_event(
                        pipeline,
                        job,
                        fingerprint,
                        "INTERRUPTED_FOR_TEST",
                        global_step=global_step,
                    )
                    return {"status": "INTERRUPTED_FOR_TEST", **job, "global_step": global_step}
                if _SIGNAL_REQUESTED:
                    slurm_id = os.environ.get("SLURM_JOB_ID")
                    if slurm_id:
                        subprocess.run(["scontrol", "requeue", slurm_id], check=True)
                    _ledger_event(
                        pipeline,
                        job,
                        fingerprint,
                        "REQUEUED",
                        global_step=global_step,
                        pass_index=pass_index,
                        cursor=cursor,
                    )
                    return {"status": "REQUEUED", **job, "global_step": global_step}
                if global_step >= total_steps:
                    break
            _save_recovery(
                latest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                fingerprint=fingerprint,
                global_step=global_step,
                pass_index=pass_index,
                cursor=cursor,
                evaluation_complete=False,
                best_score=best_score,
                best_pass=best_pass,
                without_improvement=without_improvement,
            )

        if not evaluation_complete:
            # Both arms are measured on Flickr validation after a complete
            # 1,320-step pass. Only Arm B uses this value for selection.
            if job["arm"] in {"arm_a", "arm_b"}:
                eval_config = deepcopy(config)
                eval_config["data"]["train_csv"] = eval_config["data"]["val_csv"]
                _, val_loader = build_dataloaders(eval_config)
                model.eval()
                metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    list(config["evaluation"]["k_values"]),
                )
                score = float(metrics["mean_R@1"])
                improved = score > best_score
                if job["arm"] == "arm_a" or improved:
                    best_score = score
                    best_pass = pass_index
                    without_improvement = 0
                    _atomic_torch_save(
                        {
                            "model_state": model.state_dict(),
                            "config": config,
                            "fingerprint": fingerprint.to_dict(),
                            "metrics": metrics,
                            "global_step": global_step,
                            "pass_index": pass_index,
                        },
                        save_dir / "best.pt",
                    )
                elif job["arm"] == "arm_b":
                    without_improvement += 1
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "pass_complete",
                            "pass_index": pass_index,
                            "global_step": global_step,
                            "selection_influenced": job["arm"] == "arm_b",
                            "metrics": metrics,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            evaluation_complete = True

        patience = int(config["training"]["early_stopping_patience"])
        should_stop = (
            job["arm"] == "arm_b"
            and patience >= 0
            and without_improvement >= patience
        )
        pass_index += 1
        cursor = 0
        evaluation_complete = False
        _save_recovery(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            fingerprint=fingerprint,
            global_step=global_step,
            pass_index=pass_index,
            cursor=cursor,
            evaluation_complete=evaluation_complete,
            best_score=best_score,
            best_pass=best_pass,
            without_improvement=without_improvement,
        )
        if should_stop:
            break

    summary = {
        "status": "COMPLETE",
        **job,
        **provenance,
        "global_step": global_step,
        "completed_passes": pass_index,
        "best_pass": best_pass,
        "best_flickr_validation_mean_R1": best_score,
        "selection_influenced_on_flickr_validation": job["arm"] == "arm_b",
        "flickr_test_evaluated": False,
        "fingerprint": fingerprint.digest,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(summary, save_dir / "run_summary.json")
    _ledger_event(
        pipeline,
        job,
        fingerprint,
        "COMPLETE",
        global_step=global_step,
        completed_passes=pass_index,
        best_flickr_validation_mean_R1=best_score,
        selection_influenced_on_flickr_validation=job["arm"] == "arm_b",
        flickr_test_evaluated=False,
    )
    return summary


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for job in jobs(pipeline):
        path = _path(pipeline["checkpoint_root"]) / job["run_id"] / "run_summary.json"
        if not path.is_file():
            raise RuntimeError(f"run incomplete: {job['run_id']}")
        rows.append(json.loads(path.read_text()))
    acquisition = _acquisition(pipeline)
    overlap_count = sum(
        int(value)
        for key, value in acquisition["deduplication_removed"].items()
        if key.startswith("overlap_coco_or_flickr_")
    )
    result = {
        "status": "COMPLETE",
        "runs": rows,
        "arm_a_selection_influenced": False,
        "arm_b_selection_influenced": True,
        "flickr_test_evaluated": False,
        "seed_semantics": {
            "shared": [
                "frozen selected key set",
                "shard-local ordering algorithm",
                "model and training configuration",
            ],
            "varies_by_seed": [
                "projector initialization",
                "sample order",
                "deterministic per-sample augmentation draws",
            ],
            "intentional": True,
            "interpretation": (
                "independent training replicates, not initialization-only replicates"
            ),
        },
        "acquisition_findings": {
            "minimum_side_256_rejections": int(
                acquisition["minimum_resolution_rejections"]
            ),
            "filter_denominator": int(
                acquisition["candidate_usable_before_global_dedup"]
            ),
            "minimum_side_rejection_rate_percent": (
                100
                * int(acquisition["minimum_resolution_rejections"])
                / int(acquisition["candidate_usable_before_global_dedup"])
            ),
            "unused_shards": int(
                acquisition["unused_shards_by_approved_definition"]
            ),
            "exact_coco_or_flickr_overlap_count": overlap_count,
            "heterogeneity_audited_shards": 20,
            "heterogeneity_flags": 0,
            "seeded_shard_clustering_empirically_benign": True,
            "sampling_description": (
                "deterministic random usable subset of the seed-ordered "
                "shard pool, not a uniform sample of the mirror"
            ),
        },
    }
    atomic_json(result, _path(pipeline["output_root"]) / "report.json")
    return result


def feature_cache(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Create the approved deterministic frozen-encoder cache after both arms."""
    report_path = _path(pipeline["output_root"]) / "report.json"
    if not report_path.is_file():
        raise RuntimeError("feature cache is blocked until both training arms complete")
    training_report = json.loads(report_path.read_text())
    if (
        training_report.get("status") != "COMPLETE"
        or len(training_report.get("runs", [])) != 6
        or any(row.get("status") != "COMPLETE" for row in training_report["runs"])
    ):
        raise RuntimeError("feature cache gate failed: six completed runs required")

    provenance = _hardware_guard(pipeline)
    device = provenance.pop("device")
    config = build_config(pipeline, jobs(pipeline)[0])
    cache_config = pipeline["post_training_feature_cache"]
    output = _path(cache_config["output"])
    manifest_path = output / "manifest.json"
    index_manifest = json.loads(
        (_ccroot(pipeline) / "manifests/selected_tar_index.json").read_text()
    )
    transform_hash = hash_config(cache_config["transform"])
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("source_index_sha256") == index_manifest["index_sha256"]
            and existing.get("transform_hash") == transform_hash
        ):
            return {**existing, "action": "REUSED_VERIFIED"}
        raise RuntimeError("existing feature cache does not match current provenance")

    dataset = IndexedTarDataset(
        _ccroot(pipeline) / "manifests/selected_tar_index.parquet",
        run_seed=0,
        pass_index=0,
        image_size=int(cache_config["transform"]["image"]["resize_short_side"]),
        image_mean=config["data"].get("image_mean"),
        image_std=config["data"].get("image_std"),
        interpolation=str(cache_config["transform"]["image"]["interpolation"]),
        train=False,
        deterministic_order=True,
    )
    workers = int(pipeline["training"]["dataloader_workers"])
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=image_text_collate,
        worker_init_fn=worker_init_noop,
    )
    model = build_model(config).to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    shard_rows = 65536
    shard_index = 0
    buffered_ids: list[str] = []
    buffered_image: list[torch.Tensor] = []
    buffered_text: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal shard_index
        if not buffered_ids:
            return
        path = output / f"features-{shard_index:05d}.pt"
        _atomic_torch_save(
            {
                "canonical_ids": list(buffered_ids),
                "image_features": torch.cat(buffered_image).to(torch.float16),
                "text_features": torch.cat(buffered_text).to(torch.float16),
            },
            path,
        )
        shards.append(
            {
                "path": str(path.relative_to(ROOT)),
                "rows": len(buffered_ids),
                "sha256": sha256_file(path),
            }
        )
        buffered_ids.clear()
        buffered_image.clear()
        buffered_text.clear()
        shard_index += 1

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            captions = [str(value) for value in batch["captions"]]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                image_features = model.vision_encoder(images)
                text_features = model.text_encoder(captions)
            buffered_ids.extend(str(value) for value in batch["image_paths"])
            buffered_image.append(image_features.float().cpu())
            buffered_text.append(text_features.float().cpu())
            if len(buffered_ids) >= shard_rows:
                flush()
    flush()
    result = {
        "status": "COMPLETE",
        "rows": len(dataset),
        "dtype": str(cache_config["dtype"]),
        "source_index_sha256": index_manifest["index_sha256"],
        "transform": cache_config["transform"],
        "transform_hash": transform_hash,
        "used_by_arm_a": False,
        "used_by_arm_b": False,
        "created_after_both_arms": True,
        "shards": shards,
        **provenance,
    }
    atomic_json(result, manifest_path)
    return result


def smoke(pipeline: dict[str, Any]) -> dict[str, Any]:
    provenance = _hardware_guard(pipeline)
    validate(pipeline)
    job = jobs(pipeline)[0]
    config = build_config(pipeline, job)
    device = provenance.pop("device")
    loader = _make_train_loader(pipeline, config, pass_index=0, cursor=0)
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=True)
    captions = [str(value) for value in batch["captions"]]
    image_ids = [str(value) for value in batch["image_paths"]]
    text_ids = [str(value) for value in batch["text_image_paths"]]
    model = build_model(config).to(device).train()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(images, captions)
        loss = contrastive_loss_with_memory(
            outputs["image_embeds"],
            outputs["text_embeds"],
            outputs["logit_scale"],
            image_ids,
            text_ids,
            memory=None,
        )
    loss.backward()
    result = {
        "status": "COMPLETE",
        "kind": "cc3m_tar_gpu_smoke",
        **provenance,
        "batch_size": len(image_ids),
        "unique_canonical_ids": len(set(image_ids)),
        "loss": float(loss.detach()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "precision": "native_bf16",
        "queue_enabled": False,
        "flickr_test_evaluated": False,
    }
    atomic_json(result, _path(pipeline["output_root"]) / "smoke/report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build-index", "validate", "smoke", "train", "report", "feature-cache"],
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    parser.add_argument("--stop-after-global-step", type=int)
    args = parser.parse_args()
    pipeline = _pipeline(args.pipeline)
    if args.command == "build-index":
        result = build_tar_index(pipeline)
    elif args.command == "validate":
        result = validate(pipeline)
    elif args.command == "train":
        result = train_job(
            pipeline,
            args.index,
            stop_after_global_step=args.stop_after_global_step,
        )
    elif args.command == "smoke":
        result = smoke(pipeline)
    elif args.command == "feature-cache":
        result = feature_cache(pipeline)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
