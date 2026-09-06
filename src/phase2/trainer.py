from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.transforms import build_image_transform
from src.phase15.io_utils import atomic_csv
from src.phase15.training_hard_negative_mining import _classification_training_dataset

from .bridge_model import FrozenPairBridge
from .checkpointing import PreemptionState, atomic_torch_save, checkpoint_payload, install_preemption_handlers, load_checkpoint
from .config import fingerprint
from .datasets import ClassificationHardNegativeDataset, RetrievalHardNegativeDataset, bridge_collate
from .hard_negative_loader import dataset_fingerprint, read_hard_negatives
from .losses import bridge_loss
from .status import complete_stage, managed_stage, valid_completed_run, write_status
from .schemas import RunStatus


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("training", config)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("model", config)


def _classification_dataset(root: Path, frame: pd.DataFrame, image_size: int, transform):
    datasets = {}
    for task in frame["source_task"].unique():
        dataset, _, _ = _classification_training_dataset(root, str(task), image_size, "full")
        datasets[str(task)] = dataset
    def resolve(row):
        return datasets[str(row.source_task)][int(row.sample_index)][0]
    return ClassificationHardNegativeDataset(frame, resolve, transform)


def train_bridge(
    project_root: str | Path,
    pair: dict[str, Any],
    config: dict[str, Any],
    device: str | torch.device | None = None,
    resume: bool = True,
    experiment_id: str | None = None,
    output_group: str = "bridges",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    training, model_cfg = _training_config(config), _model_config(config)
    config_id = str(pair["pair_id"])
    experiment_id = experiment_id or config_id
    run_dir = root / f"results/phase2/{output_group}/runs" / experiment_id
    cfg_fingerprint = fingerprint({key: value for key, value in config.items() if key != "_config_path"})
    if valid_completed_run(run_dir, cfg_fingerprint):
        import json
        summary = json.loads((run_dir / "summary.json").read_text())
        return {**summary, "status": "skipped_complete", "run_dir": str(run_dir)}
    data_root = root / "results/phase2/data"
    data_fp = dataset_fingerprint(data_root)
    maximum = training.get("max_samples_per_source")
    negative_config_id = str(training.get("negative_config_id", config_id))
    retrieval = read_hard_negatives(data_root / "retrieval_train.parquet", kind="retrieval", config_id=negative_config_id, max_rows=maximum)
    classification = read_hard_negatives(data_root / "classification_train.parquet", kind="classification", config_id=negative_config_id, max_rows=maximum)
    if retrieval.empty or classification.empty:
        raise ValueError(
            f"Training-negative pool {negative_config_id!r} is empty "
            f"(retrieval={len(retrieval)}, classification={len(classification)})"
        )
    image_size = int(training.get("image_size", 224))
    transform = build_image_transform(image_size, train=True)
    dataset = ConcatDataset([
        RetrievalHardNegativeDataset(retrieval, transform, str(training.get("negative_source", "hard"))),
        _classification_dataset(root, classification, image_size, transform),
    ])
    loader = DataLoader(
        dataset, batch_size=int(training.get("batch_size", 16)), shuffle=True,
        num_workers=int(training.get("num_workers", 4)), pin_memory=target.type == "cuda",
        collate_fn=bridge_collate, drop_last=True,
    )
    bridge = FrozenPairBridge.from_selected_pair(
        pair, target, bridge_dim=int(model_cfg.get("bridge_dim", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 4)), layers=int(model_cfg.get("layers", 1)),
        feedforward_dim=int(model_cfg.get("feedforward_dim", 512)), dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "masked_mean")),
    )
    optimizer = torch.optim.AdamW(bridge.trainable_parameters(), lr=float(training.get("learning_rate", 2e-4)), weight_decay=float(training.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training.get("epochs", 10))))
    amp_enabled = bool(training.get("amp", True)) and target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_dir = root / f"results/phase2/{output_group}/checkpoints" / experiment_id
    last_checkpoint, best_checkpoint = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    start_epoch, global_step, best_loss, patience = 0, 0, math.inf, 0
    if resume and last_checkpoint.exists():
        state = load_checkpoint(last_checkpoint, bridge, optimizer, scheduler, scaler, target)
        if state.get("config_fingerprint") != cfg_fingerprint or state.get("dataset_fingerprint") != data_fp:
            raise RuntimeError("Resume checkpoint fingerprint mismatch")
        start_epoch, global_step, best_loss = int(state["epoch"]) + 1, int(state["global_step"]), float(state["best_loss"])
    install_preemption_handlers()
    metrics: list[dict[str, Any]] = []
    with managed_stage(run_dir, "bridge_training", config, root):
        for epoch in range(start_epoch, int(training.get("epochs", 10))):
            bridge.train(); optimizer.zero_grad(set_to_none=True); running = 0.0; batches = 0
            accumulation = int(training.get("gradient_accumulation", 1))
            gradient_checked = False
            for batch_index, batch in enumerate(loader):
                positive_images = batch["positive_images"].to(target, non_blocking=True)
                negative_images = batch["negative_images"].to(target, non_blocking=True)
                with torch.amp.autocast(target.type, enabled=amp_enabled):
                    positive = bridge(positive_images, batch["positive_captions"])
                    negative = bridge(negative_images, batch["negative_captions"])
                    losses = bridge_loss(
                        positive["scores"], negative["scores"], positive["features"], negative["features"],
                        rank_weight=float(training.get("rank_weight", 1.0)),
                        contrastive_weight=float(training.get("contrastive_weight", 0.0)),
                        margin=float(training.get("ranking_margin", 0.2)),
                    )
                    loss = losses["loss"] / accumulation
                scaler.scale(loss).backward()
                if not gradient_checked:
                    bridge.validate_gradients(); gradient_checked = True
                if (batch_index + 1) % accumulation == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); global_step += 1
                running += float(losses["loss"].detach()); batches += 1
                if PreemptionState.requested:
                    state = checkpoint_payload(bridge, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step, best_loss=best_loss, config_fingerprint=cfg_fingerprint, dataset_fingerprint=data_fp, world_size=1)
                    atomic_torch_save(state, last_checkpoint)
                    write_status(run_dir, RunStatus.INTERRUPTED, stage="bridge_training", signal=PreemptionState.signal_number)
                    raise KeyboardInterrupt("Slurm pre-termination checkpoint completed")
            epoch_loss = running / max(1, batches)
            scheduler.step()
            metrics.append({"epoch": epoch, "loss": epoch_loss, "learning_rate": scheduler.get_last_lr()[0], "global_step": global_step})
            atomic_csv(pd.DataFrame(metrics), run_dir / "metrics_by_epoch.csv")
            state = checkpoint_payload(bridge, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step, best_loss=min(best_loss, epoch_loss), config_fingerprint=cfg_fingerprint, dataset_fingerprint=data_fp, world_size=1)
            atomic_torch_save(state, last_checkpoint)
            if epoch_loss < best_loss:
                best_loss, patience = epoch_loss, 0
                state["best_loss"] = best_loss; atomic_torch_save(state, best_checkpoint)
            else:
                patience += 1
            if (epoch + 1) % int(training.get("checkpoint_interval", 2)) == 0:
                atomic_torch_save(state, checkpoint_dir / f"epoch_{epoch:03d}.pt")
            if patience >= int(training.get("early_stopping_patience", 3)):
                break
        summary = {"status": "COMPLETED", "pair_id": config_id, "training_negative_config_id": negative_config_id, "best_loss": best_loss, "epochs_completed": len(metrics), "global_step": global_step, "best_checkpoint": str(best_checkpoint), "config_fingerprint": cfg_fingerprint, "dataset_fingerprint": data_fp, "backbones_frozen": True}
        complete_stage(run_dir, "bridge_training", summary)
    return summary
