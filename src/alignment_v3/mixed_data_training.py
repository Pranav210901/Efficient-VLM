from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.alignment_v3.cc3m_training import (
    _atomic_torch_save,
    _git_sha,
    _hardware_guard,
)
from src.alignment_v3.distillation import (
    TeacherCache,
    distillation_loss,
    save_teacher_cache,
)
from src.alignment_v3.fingerprint import (
    Fingerprint,
    code_fingerprint,
    hash_config,
    sha256_file,
    write_fingerprint,
)
from src.alignment_v3.model import DINO_VISION_MODELS, build_model
from src.alignment_v3.references import build_reference
from src.alignment_v3.training import (
    _optimizer_parameters,
    _precision,
    _restore_rng,
    _rng_state,
    _scheduler,
)
from src.data import build_dataloaders
from src.data.collate import image_text_collate
from src.data.mixed_coco_cc3m import (
    LogicalCursorSampler,
    MixedCocoCC3MDataset,
)
from src.models.text_encoders import TEXT_MODEL_REGISTRY
from src.phase15.io_utils import atomic_csv, atomic_json
from src.training.evaluate import evaluate_model
from src.training.losses import contrastive_loss_with_memory
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = "configs/mixed_data_training/pipeline.yaml"
CODE_PATHS = [
    "src/alignment_v3/mixed_data_training.py",
    "src/data/mixed_coco_cc3m.py",
    "src/alignment_v3/training.py",
    "src/alignment_v3/model.py",
    "src/training/losses.py",
]


def pipeline(path: str | Path = DEFAULT_PIPELINE) -> dict[str, Any]:
    value = Path(path)
    return load_config(value if value.is_absolute() else ROOT / value)


def rooted(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def jobs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "arm": str(arm["id"]),
            "seed": int(seed),
            "run_id": f"{arm['id']}__seed_{int(seed)}",
        }
        for index, (arm, seed) in enumerate(
            (arm, seed)
            for arm in spec["training"]["arms"]
            for seed in spec["training"]["seeds"]
        )
    ]


def selected_job(spec: dict[str, Any], index: int | None) -> dict[str, Any]:
    value = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else index
    return jobs(spec)[int(value)]


def arm_spec(spec: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(value for value in spec["training"]["arms"] if value["id"] == arm_id)


def build_config(spec: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(load_config(rooted(spec["base_config"])))
    arm = arm_spec(spec, job["arm"])
    config["seed"] = int(job["seed"])
    config["experiment_id"] = "mixed_data_training"
    config["run_id"] = str(job["run_id"])
    config["data"]["image_size"] = int(spec["data"]["image_size"])
    config["data"]["num_workers"] = int(spec["training"]["dataloader_workers"])
    config["data"]["val_csv"] = str(spec["data"]["flickr_validation_csv"])
    config["training"].update(
        {
            "batch_size": int(spec["training"]["batch_size"]),
            "drop_last": False,
            "lr": float(spec["training"]["resolved_lr"]),
            "max_optimizer_steps": int(arm["total_optimizer_steps"]),
            "epochs": int(spec["training"]["passes"]),
            "warmup_fraction": float(spec["training"]["warmup_fraction"]),
            "precision": "bf16",
            "memory_queue_size": 0,
            "queue_mode": "none",
            "save_dir": str(
                (rooted(spec["checkpoint_root"]) / job["run_id"]).relative_to(ROOT)
            ),
        }
    )
    config["recipe"]["loss_type"] = "infonce_no_queue"
    config["recipe"]["memory_queue_size"] = 0
    config["recipe"]["distillation"] = True
    config["distillation"]["strength"] = 1.0
    config["provenance"]["mixed_data_training"] = {
        "arm": job["arm"],
        "pass_unit": "image",
        "all_coco_captions_are_positive_rows": True,
        "cc3m_captions_per_image": 1,
        "uniform_image_sampling": True,
        "passes": int(spec["training"]["passes"]),
        "images_per_pass": int(arm["images_per_pass"]),
        "steps_per_pass": int(arm["optimizer_steps_per_pass"]),
        "total_steps": int(arm["total_optimizer_steps"]),
        "lr_tuned_for_mixture": False,
        "flickr_test_evaluated": False,
    }
    return config


def fingerprint(config: dict[str, Any], spec: dict[str, Any]) -> Fingerprint:
    return Fingerprint(
        config_hash=hash_config(config),
        dataset_split_hash=hash_config(
            {
                "cc3m_index": sha256_file(rooted(spec["data"]["cc3m_index"])),
                "coco_csv": (
                    sha256_file(rooted(spec["data"]["coco_csv"]))
                    if config["provenance"]["mixed_data_training"]["arm"]
                    == "mixed_coco_cc3m"
                    else None
                ),
            }
        ),
        model_checkpoint_id=(
            f"{DINO_VISION_MODELS[config['model']['vision_encoder']]}+"
            f"{TEXT_MODEL_REGISTRY[config['model']['text_encoder']]}+C4+"
            f"{spec['teacher']['id']}"
        ),
        cache_version="mixed-mobileclip2-v1",
        preprocessing_hash=hash_config({"image_size": config["data"]["image_size"]}),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
        seed=int(config["seed"]),
    )


class CC3MTeacherDataset(Dataset[dict[str, object]]):
    def __init__(self, index_path: Path, transform: Any) -> None:
        self.data = pq.read_table(
            index_path,
            columns=["canonical_id", "tar_path", "offset", "size", "caption", "raw_sha256"],
            memory_map=True,
        ).to_pydict()
        self.transform = transform
        self._handles: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.data["canonical_id"])

    def __getitem__(self, index: int) -> dict[str, object]:
        path = str(self.data["tar_path"][index])
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = open(path, "rb", buffering=0)
            self._handles[path] = handle
        offset, size = int(self.data["offset"][index]), int(self.data["size"][index])
        handle.seek(offset)
        payload = handle.read(size)
        if hashlib.sha256(payload).hexdigest() != str(self.data["raw_sha256"][index]):
            raise RuntimeError(f"teacher-cache source hash mismatch at row {index}")
        with Image.open(io.BytesIO(payload)) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "caption": str(self.data["caption"][index]),
            "image_path": str(self.data["canonical_id"][index]),
        }


def build_cc3m_teacher_cache(spec: dict[str, Any]) -> dict[str, Any]:
    target = rooted(spec["teacher"]["cc3m_cache"])
    index_path = rooted(spec["data"]["cc3m_index"])
    reference = build_reference(str(spec["teacher"]["id"]))
    expected = Fingerprint(
        config_hash=hash_config(spec["teacher"]),
        dataset_split_hash=sha256_file(index_path),
        model_checkpoint_id=(
            f"{spec['teacher']['registry_tag']}:{spec['teacher']['pretrained']}:"
            f"open_clip-{spec['teacher']['open_clip_version']}"
        ),
        cache_version="cc3m-mobileclip2-v1",
        preprocessing_hash=hash_config({"native_transform": repr(reference.preprocess)}),
        code_version=code_fingerprint(ROOT, CODE_PATHS),
    )
    metadata = target.with_suffix(".metadata.json")
    if target.is_file() and metadata.is_file():
        value = json.loads(metadata.read_text())
        if (
            value.get("fingerprint_digest") == expected.digest
            and value.get("content_sha256") == sha256_file(target)
        ):
            return {"status": "SKIPPED_FRESH", **value}
        raise RuntimeError("existing CC3M teacher cache has incompatible provenance")
    hardware = _hardware_guard(spec)
    device = hardware.pop("device")
    reference = reference.to(device).eval()
    dataset = CC3MTeacherDataset(index_path, reference.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(spec["training"]["dataloader_workers"]),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=image_text_collate,
    )
    images, texts, image_paths, text_paths, captions = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch_captions = [str(x) for x in batch["captions"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                images.append(reference.encode_image(batch["images"].to(device)).cpu())
                texts.append(reference.encode_text(batch_captions).cpu())
            image_paths.extend(str(x) for x in batch["image_paths"])
            text_paths.extend(str(x) for x in batch["text_image_paths"])
            captions.extend(batch_captions)
    result = save_teacher_cache(
        target,
        image_embeddings=torch.cat(images),
        text_embeddings=torch.cat(texts),
        image_paths=image_paths,
        text_image_paths=text_paths,
        captions=captions,
        fingerprint=expected,
        teacher_id=str(spec["teacher"]["id"]),
        logit_scale=reference.logit_scale_value,
    )
    if int(result["image_rows"]) != int(spec["teacher"]["cc3m_cache_rows"]):
        raise RuntimeError("CC3M teacher cache row count mismatch")
    return {"status": "COMPLETE", **hardware, **result}


class CompositeTeacherCache:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.caches = [TeacherCache(path) for path in paths]
        scales = [float(cache.metadata["logit_scale"]) for cache in self.caches]
        if max(scales) - min(scales) > 1e-6:
            raise RuntimeError("teacher-cache logit scales disagree")
        self.metadata = {"logit_scale": scales[0]}

    def lookup(
        self,
        image_paths: Sequence[str],
        text_image_paths: Sequence[str],
        captions: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_rows, text_rows = [], []
        for image_path in image_paths:
            found = None
            for cache in self.caches:
                try:
                    found = cache.lookup([image_path], [], [], device)[0][0]
                    break
                except KeyError:
                    pass
            if found is None:
                raise KeyError(f"no teacher image target for {image_path}")
            image_rows.append(found)
        for path, caption in zip(text_image_paths, captions):
            found = None
            for cache in self.caches:
                try:
                    found = cache.lookup([], [path], [caption], device)[1][0]
                    break
                except KeyError:
                    pass
            if found is None:
                raise KeyError(f"no teacher text target for {path}")
            text_rows.append(found)
        return torch.stack(image_rows), torch.stack(text_rows)


def _loader(
    spec: dict[str, Any],
    config: dict[str, Any],
    *,
    pass_index: int,
    cursor: int,
) -> DataLoader:
    mixed = config["provenance"]["mixed_data_training"]["arm"] == "mixed_coco_cc3m"
    dataset = MixedCocoCC3MDataset(
        cc3m_index=rooted(spec["data"]["cc3m_index"]),
        coco_csv=rooted(spec["data"]["coco_csv"]) if mixed else None,
        arm_id=config["provenance"]["mixed_data_training"]["arm"],
        run_seed=int(config["seed"]),
        pass_index=pass_index,
        image_size=int(config["data"]["image_size"]),
        train=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=LogicalCursorSampler(len(dataset), cursor),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
        collate_fn=image_text_collate,
    )


_STOP = False


def _request_stop(_: int, __: Any) -> None:
    global _STOP
    _STOP = True


def _save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    fp: Fingerprint,
    global_step: int,
    pass_index: int,
    cursor: int,
) -> None:
    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _rng_state(),
            "config": config,
            "fingerprint": fp.to_dict(),
            "global_step": global_step,
            "pass_index": pass_index,
            "cursor": cursor,
        },
        path,
    )


def train_job(spec: dict[str, Any], index: int | None) -> dict[str, Any]:
    global _STOP
    _STOP = False
    hardware = _hardware_guard(spec)
    device = hardware.pop("device")
    job = selected_job(spec, index)
    config = build_config(spec, job)
    fp = fingerprint(config, spec)
    save_dir = rooted(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")
    write_fingerprint(save_dir / "fingerprint.json", fp)
    summary_path = save_dir / "run_summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text())
        if prior.get("fingerprint") == fp.digest:
            return {"status": "SKIPPED_FRESH", **job}
        raise RuntimeError("completed directory belongs to another fingerprint")
    set_seed(int(config["seed"]), deterministic=False)
    model = build_model(config).to(device)
    parameters = model.parameter_summary()
    if int(parameters["params_trainable_inference"]) != 2_730_628:
        raise RuntimeError(f"C4 parameter contract failed: {parameters}")
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, float(config["training"]["weight_decay"])),
        lr=float(config["training"]["lr"]),
        betas=tuple(config["training"]["betas"]),
        fused=True,
    )
    scheduler = _scheduler(
        optimizer,
        int(config["training"]["max_optimizer_steps"]),
        float(config["training"]["warmup_fraction"]),
    )
    amp, dtype, scale = _precision(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=scale)
    teacher = CompositeTeacherCache(
        [
            rooted(spec["teacher"]["coco_cache"]),
            rooted(spec["teacher"]["cc3m_cache"]),
        ]
    )
    latest = save_dir / "latest.pt"
    global_step = pass_index = cursor = 0
    if latest.is_file():
        state = torch.load(latest, map_location=device, weights_only=False)
        if Fingerprint.from_dict(state["fingerprint"]).digest != fp.digest:
            raise RuntimeError("refusing cross-fingerprint resume")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        _restore_rng(state["rng_state"])
        global_step, pass_index, cursor = (
            int(state["global_step"]),
            int(state["pass_index"]),
            int(state["cursor"]),
        )
    signal.signal(signal.SIGUSR1, _request_stop)
    metrics_path = save_dir / "training.jsonl"
    started = time.perf_counter()
    passes = int(spec["training"]["passes"])
    while pass_index < passes:
        loader = _loader(spec, config, pass_index=pass_index, cursor=cursor)
        model.train()
        running = {"loss": 0.0, "contrastive": 0.0, "distillation": 0.0, "batches": 0}
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            captions = [str(x) for x in batch["captions"]]
            image_ids = [str(x) for x in batch["image_paths"]]
            text_ids = [str(x) for x in batch["text_image_paths"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=dtype):
                outputs = model(images, captions)
                contrastive = contrastive_loss_with_memory(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    outputs["logit_scale"],
                    image_ids,
                    text_ids,
                    memory=None,
                )
                ti, tt = teacher.lookup(image_ids, text_ids, captions, device)
                auxiliary, _ = distillation_loss(
                    outputs,
                    ti,
                    tt,
                    teacher_scale=float(teacher.metadata["logit_scale"]),
                    temperature=float(config["distillation"]["temperature"]),
                    image_cosine_weight=float(config["distillation"]["image_cosine_weight"]),
                    text_cosine_weight=float(config["distillation"]["text_cosine_weight"]),
                    kl_weight=float(config["distillation"]["kl_weight"]),
                )
                loss = contrastive + float(config["distillation"]["strength"]) * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))
            global_step += 1
            cursor += len(image_ids)
            running["loss"] += float(loss.detach())
            running["contrastive"] += float(contrastive.detach())
            running["distillation"] += float(auxiliary.detach())
            running["batches"] += 1
            if (
                global_step % int(spec["training"]["checkpoint_interval_steps"]) == 0
                or _STOP
            ):
                _save_state(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    fp=fp,
                    global_step=global_step,
                    pass_index=pass_index,
                    cursor=cursor,
                )
            if _STOP:
                slurm_id = os.environ.get("SLURM_JOB_ID")
                if slurm_id:
                    subprocess.run(["scontrol", "requeue", slurm_id], check=True)
                return {"status": "REQUEUED", **job, "global_step": global_step}
        pass_index += 1
        cursor = 0
        _save_state(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            fp=fp,
            global_step=global_step,
            pass_index=pass_index,
            cursor=cursor,
        )
        snapshot = save_dir / f"pass_{pass_index:02d}.pt"
        if snapshot.exists():
            saved = torch.load(snapshot, map_location="cpu", weights_only=False)
            if int(saved["pass_index"]) != pass_index:
                raise RuntimeError("immutable pass snapshot mismatch")
        else:
            os.link(latest, snapshot)
        row = {
            "kind": "pass_complete",
            "pass": pass_index,
            "global_step": global_step,
            **{key: value / max(1, running["batches"]) for key, value in running.items() if key != "batches"},
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "status": "COMPLETE",
        **job,
        **hardware,
        "completed_passes": pass_index,
        "global_step": global_step,
        "fingerprint": fp.digest,
        "parameter_summary": parameters,
        "wall_seconds": time.perf_counter() - started,
        "flickr_test_evaluated": False,
    }
    atomic_json(summary, summary_path)
    return summary


def evaluate_snapshot(spec: dict[str, Any], index: int | None) -> dict[str, Any]:
    value = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else index
    job_index, pass_offset = divmod(int(value), int(spec["training"]["passes"]))
    job = jobs(spec)[job_index]
    pass_number = pass_offset + 1
    config = build_config(spec, job)
    save_dir = rooted(config["training"]["save_dir"])
    snapshot = save_dir / f"pass_{pass_number:02d}.pt"
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(state["model_state"], strict=True)
    hardware = _hardware_guard(spec)
    device = hardware.pop("device")
    model = model.to(device).eval()
    rows = {}
    datasets = {
        "coco_dev": "data/splits/alignment_v3_recovery/coco_dev_5000.csv",
        "flickr_validation": str(spec["data"]["flickr_validation_csv"]),
    }
    for name, csv_path in datasets.items():
        eval_config = deepcopy(config)
        eval_config["data"]["train_csv"] = csv_path
        eval_config["data"]["val_csv"] = csv_path
        eval_config["data"]["group_by_image"] = True
        eval_config["data"]["val_captions_per_image"] = None
        _, loader = build_dataloaders(eval_config)
        rows[name] = evaluate_model(
            model, loader, device, list(config["evaluation"]["k_values"])
        )
    result = {
        "status": "COMPLETE",
        **job,
        "pass": pass_number,
        "global_step": int(state["global_step"]),
        "metrics": rows,
        "flickr_test_evaluated": False,
        **hardware,
    }
    atomic_json(result, save_dir / "evaluations" / f"pass_{pass_number:02d}.json")
    return result


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    selection = load_config(rooted(spec["frozen_selection"]))
    if selection.get("status") != "FROZEN_FOR_DATA_STAGE" or selection.get(
        "selected_candidate"
    ) != "C4":
        raise RuntimeError("C4 frozen selection is missing")
    if rooted(spec["data"]["flickr_test_csv"]).as_posix() in json.dumps(spec["reporting"]):
        if spec["reporting"].get("evaluate_flickr_test") is not False:
            raise RuntimeError("Flickr test must remain sealed")
    table = pq.read_table(rooted(spec["data"]["cc3m_index"]), columns=["canonical_id"])
    if table.num_rows != int(spec["data"]["cc3m_unique_images"]):
        raise RuntimeError("CC3M index row count changed")
    coco = pd.read_csv(rooted(spec["data"]["coco_csv"]))
    coco_images = int(coco["image_path"].nunique())
    if coco_images != int(spec["data"]["coco_unique_images"]):
        raise RuntimeError(f"COCO unique-image count changed: {coco_images}")
    for arm in spec["training"]["arms"]:
        expected = math.ceil(int(arm["images_per_pass"]) / int(spec["training"]["batch_size"]))
        if expected != int(arm["optimizer_steps_per_pass"]):
            raise RuntimeError(f"{arm['id']} step arithmetic mismatch")
        if expected * int(spec["training"]["passes"]) != int(arm["total_optimizer_steps"]):
            raise RuntimeError(f"{arm['id']} total-step mismatch")
    configs = [build_config(spec, job) for job in jobs(spec)]
    if any(config["recipe"]["image_token_aggregation"] != "transformer_128" for config in configs):
        raise RuntimeError("C4 token aggregation changed")
    result = {
        "status": "READY",
        "training_jobs": 6,
        "evaluation_jobs": 144,
        "teacher_cache_required": not rooted(spec["teacher"]["cc3m_cache"]).is_file(),
        "arms": spec["training"]["arms"],
        "caption_semantics": {
            "pass_unit": "image",
            "coco": "all available captions become positive text rows",
            "cc3m": "one caption per image",
            "source_weighting": "uniform per image, not per caption",
        },
        "flickr_test_sealed": True,
    }
    atomic_json(result, rooted(spec["output_root"]) / "manifests/design.json")
    return result


def report(spec: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for job in jobs(spec):
        save_dir = rooted(spec["checkpoint_root"]) / job["run_id"]
        summary = json.loads((save_dir / "run_summary.json").read_text())
        for pass_number in range(1, int(spec["training"]["passes"]) + 1):
            value = json.loads(
                (save_dir / "evaluations" / f"pass_{pass_number:02d}.json").read_text()
            )
            rows.append(
                {
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "pass": pass_number,
                    "global_step": value["global_step"],
                    "coco_dev_mean_R1": value["metrics"]["coco_dev"]["mean_R@1"],
                    "flickr_validation_mean_R1": value["metrics"]["flickr_validation"]["mean_R@1"],
                    "completed_training": summary["status"] == "COMPLETE",
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, rooted(spec["output_root"]) / "report/per_pass.csv")
    selections = []
    for (arm, seed), group in frame.groupby(["arm", "seed"]):
        coco = group.loc[group["coco_dev_mean_R1"].idxmax()]
        flickr = group.loc[group["flickr_validation_mean_R1"].idxmax()]
        selections.append(
            {
                "arm": arm,
                "seed": int(seed),
                "coco_selected_pass": int(coco["pass"]),
                "coco_selected_flickr_R1": float(coco["flickr_validation_mean_R1"]),
                "flickr_selected_pass": int(flickr["pass"]),
                "flickr_selected_R1": float(flickr["flickr_validation_mean_R1"]),
                "value_at_pass_24": float(
                    group.loc[group["pass"] == 24, "flickr_validation_mean_R1"].iloc[0]
                ),
                "peaked_at_ceiling": int(flickr["pass"]) == 24,
            }
        )
    selected = pd.DataFrame(selections)
    atomic_csv(selected, rooted(spec["output_root"]) / "report/per_seed_selection.csv")
    summary = (
        selected.groupby("arm", as_index=False)
        .agg(
            coco_selected_mean=("coco_selected_flickr_R1", "mean"),
            flickr_selected_mean=("flickr_selected_R1", "mean"),
            flickr_selected_sd=("flickr_selected_R1", "std"),
            flickr_selected_min=("flickr_selected_R1", "min"),
            flickr_selected_max=("flickr_selected_R1", "max"),
            seeds_peaking_at_pass24=("peaked_at_ceiling", "sum"),
        )
    )
    summary["selection_gain_pp"] = (
        summary["flickr_selected_mean"] - summary["coco_selected_mean"]
    ) * 100
    atomic_csv(summary, rooted(spec["output_root"]) / "report/summary.csv")
    result = {
        "status": "COMPLETE",
        "arms": summary.to_dict("records"),
        "latency_changed": False,
        "architecture": "frozen C4",
        "flickr_test_evaluated": False,
        "cc3m_qualification": (
            "pixparse/cc3m-wds revision "
            "46f3d69f840e59d77d52e8decfe5baec97e94c7f"
        ),
    }
    atomic_json(result, rooted(spec["output_root"]) / "report/report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["validate", "teacher-cache", "train", "evaluate", "report"],
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    spec = pipeline(args.pipeline)
    functions = {
        "validate": lambda: validate(spec),
        "teacher-cache": lambda: build_cc3m_teacher_cache(spec),
        "train": lambda: train_job(spec, args.index),
        "evaluate": lambda: evaluate_snapshot(spec, args.index),
        "report": lambda: report(spec),
    }
    print(json.dumps(functions[args.command](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
