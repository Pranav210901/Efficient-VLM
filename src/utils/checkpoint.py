from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    *,
    scheduler: object | None = None,
    scaler: object | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,  # type: ignore[attr-defined]
            "scaler_state": scaler.state_dict() if scaler is not None else None,  # type: ignore[attr-defined]
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def load_checkpoint_metrics(path: str | Path) -> dict[str, float]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    return {str(key): float(value) for key, value in checkpoint.get("metrics", {}).items()}
