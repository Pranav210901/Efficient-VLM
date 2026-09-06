from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.data import build_dataloaders
from src.models import BLFVisionLanguageModel
from src.training.evaluate import evaluate_model
from src.training.losses import CrossBatchMemory, contrastive_loss_with_memory
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device
from src.utils.logging import CSVLogger, count_parameters
from src.utils.config import save_config
from src.utils.seed import set_seed


def build_model_from_config(config: dict[str, Any]) -> BLFVisionLanguageModel:
    model_cfg = config["model"]
    return BLFVisionLanguageModel(
        vision_encoder_name=str(model_cfg["vision_encoder"]),
        text_encoder_name=str(model_cfg["text_encoder"]),
        use_local_blf=bool(model_cfg.get("use_local_blf", True)),
        use_global_blf=bool(model_cfg.get("use_global_blf", True)),
        fusion_type=str(model_cfg.get("fusion_type", "concat_mlp")),
        shared_dim=int(model_cfg.get("shared_dim", 256)),
        local_dim=int(model_cfg.get("local_dim", 128)),
        global_dim=int(model_cfg.get("global_dim", 128)),
        fusion_dim=int(model_cfg.get("fusion_dim", 512)),
        projection_hidden_dim=int(model_cfg.get("projection_hidden_dim", 512)),
        projection_dropout=float(model_cfg.get("projection_dropout", 0.1)),
        freeze_vision=bool(model_cfg.get("freeze_vision", True)),
        freeze_text=bool(model_cfg.get("freeze_text", True)),
        pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
        pretrained_text=bool(model_cfg.get("pretrained_text", True)),
        vision_backend=str(model_cfg.get("vision_backend", "auto")),
        text_prefix=model_cfg.get("text_prefix"),
        projection_type=str(model_cfg.get("projection_type", "mlp")),
    )


def _optimizer_parameters(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "logit_scale" in name or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_fraction: float,
    scheduler_name: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(total_steps - 1, max(0, int(round(total_steps * warmup_fraction))))

    def scale(step: int) -> float:
        if scheduler_name == "constant":
            return 1.0
        if scheduler_name != "cosine":
            raise ValueError(f"unsupported scheduler {scheduler_name!r}")
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def checkpoint_resume_path(save_dir: Path) -> Path | None:
    latest_path = save_dir / "latest.pt"
    if latest_path.exists():
        return latest_path
    best_path = save_dir / "best.pt"
    if best_path.exists():
        return best_path
    return None


def trim_metrics_after_epoch(path: Path, epoch: int) -> None:
    if not path.exists():
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "epoch" not in fieldnames:
            return
        rows = [row for row in reader if int(float(row.get("epoch", 0))) <= epoch]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_from_config(config: dict[str, Any], resume: bool = False) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    device = get_device(str(config.get("device", "auto")))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model_from_config(config).to(device)
    param_info = count_parameters(model)
    print(f"Device: {device}")
    print(f"Vision encoder: {config['model']['vision_encoder']} -> dim {model.vision_encoder.output_dim}")
    print(f"Text encoder: {config['model']['text_encoder']} -> dim {model.text_encoder.output_dim}")
    print(f"Parameters: total={param_info['params_total']:.0f}, trainable={param_info['params_trainable']:.0f}")

    train_cfg = config["training"]
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, weight_decay),
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=tuple(float(value) for value in train_cfg.get("betas", [0.9, 0.999])),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger = CSVLogger(save_dir / "metrics.csv")
    best_r1 = -1.0
    best_metrics: dict[str, float] = {}
    k_values = list(config.get("evaluation", {}).get("k_values", [1, 5, 10]))
    start_epoch = 1
    total_epochs = int(train_cfg.get("epochs", 5))
    scheduler = _cosine_scheduler(
        optimizer,
        total_epochs,
        max(1, len(train_loader)),
        float(train_cfg.get("warmup_fraction", 0.0)),
        str(train_cfg.get("scheduler", "constant")),
    )
    memory = CrossBatchMemory(int(train_cfg.get("memory_queue_size", 0)))
    gradient_clip_norm = float(train_cfg.get("gradient_clip_norm", 0.0))
    selection_metric = str(train_cfg.get("selection_metric", "i2t_R@1"))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", -1))
    epochs_without_improvement = 0
    best_epoch = 0
    completed_epoch = 0

    if resume:
        resume_path = checkpoint_resume_path(save_dir)
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            if checkpoint.get("optimizer_state") is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if checkpoint.get("scaler_state") is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metrics = {**checkpoint.get("metrics", {}), **param_info}
            best_r1 = float(best_metrics.get(selection_metric, -1.0))
            best_path = save_dir / "best.pt"
            if best_path.exists():
                best_checkpoint = torch.load(best_path, map_location="cpu")
                best_metrics = {**best_checkpoint.get("metrics", {}), **param_info}
                best_r1 = float(best_metrics.get(selection_metric, best_r1))
                best_epoch = int(best_checkpoint.get("epoch", 0))
            trim_metrics_after_epoch(save_dir / "metrics.csv", start_epoch - 1)
            print(f"Resuming from {resume_path} at epoch {start_epoch}/{total_epochs}")

    if start_epoch > total_epochs:
        final_metrics = evaluate_model(model, val_loader, device, k_values=k_values)
        final_metrics.update(param_info)
        save_checkpoint(save_dir / "final.pt", model, optimizer, total_epochs, config, final_metrics, scheduler=scheduler, scaler=scaler)
        summary = {
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "completed_epoch": total_epochs,
            "stopped_early": False,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
        }
        (save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if not bool(train_cfg.get("keep_latest", True)):
            (save_dir / "latest.pt").unlink(missing_ok=True)
        return best_metrics or final_metrics

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            images = batch["images"].to(device)
            captions = list(batch["captions"])
            image_ids = [str(value) for value in batch["image_paths"]]
            text_image_ids = [str(value) for value in batch.get("text_image_paths", batch["image_paths"])]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images, captions)
                loss = contrastive_loss_with_memory(
                    outputs["image_embeds"],
                    outputs["text_embeds"],
                    outputs["logit_scale"],
                    image_ids,
                    text_image_ids,
                    memory,
                )
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            memory.enqueue(outputs["image_embeds"], outputs["text_embeds"], image_ids, text_image_ids)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))
            running += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader))
        metrics = evaluate_model(model, val_loader, device, k_values=k_values)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "logit_scale": float(model.logit_scale.exp().detach().cpu().clamp(max=100.0).item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **param_info,
        }
        logger.write(row)
        print(row)
        latest_metrics = {**metrics, **param_info}
        save_checkpoint(save_dir / "latest.pt", model, optimizer, epoch, config, latest_metrics, scheduler=scheduler, scaler=scaler)
        if selection_metric not in metrics:
            raise KeyError(f"selection metric {selection_metric!r} is not produced by evaluation")
        if metrics[selection_metric] > best_r1:
            best_r1 = metrics[selection_metric]
            best_metrics = {**metrics, **param_info}
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, config, best_metrics, scheduler=scheduler, scaler=scaler)
        else:
            epochs_without_improvement += 1
        completed_epoch = epoch
        if (
            metrics.get(selection_metric, 0.0) <= best_r1
            and epochs_without_improvement > 0
            and early_stopping_patience >= 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    final_metrics = evaluate_model(model, val_loader, device, k_values=k_values)
    final_metrics.update(param_info)
    save_checkpoint(save_dir / "final.pt", model, optimizer, completed_epoch, config, final_metrics, scheduler=scheduler, scaler=scaler)
    summary = {
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "stopped_early": completed_epoch < total_epochs,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
    }
    (save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not bool(train_cfg.get("keep_latest", True)):
        (save_dir / "latest.pt").unlink(missing_ok=True)
    return best_metrics or final_metrics
