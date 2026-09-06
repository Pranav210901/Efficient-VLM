from __future__ import annotations

from pathlib import Path
import math
import os
import time

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from .checkpointing import atomic_torch_save, checkpoint_payload, install_preemption_handlers, load_checkpoint, PreemptionState
from .config import fingerprint
from .dense_teacher import DenseTaskConditionedTeacher, teacher_regularisation
from .distributed import cleanup_distributed, initialise_distributed, rank_zero, reduce_mean, sampler_for, seed_rank
from .status import complete_stage, managed_stage, write_status
from .schemas import RunStatus


def _wrap_dense_teacher(model: DenseTaskConditionedTeacher, context):
    if not context.distributed:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        find_unused_parameters=True,
    )


def _task_codes(values: pd.Series) -> tuple[torch.Tensor, list[str]]:
    names = sorted(values.astype(str).unique().tolist())
    categorical = pd.Categorical(values.astype(str), categories=names)
    if (categorical.codes < 0).any():
        raise ValueError("Dense-teacher task labels contain missing values")
    return torch.tensor(categorical.codes, dtype=torch.long), names


def _teacher_evaluation_data(root: Path, depth: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    paths = sorted((root / "results/phase2/reranking/runs").glob("*/per_sample_results.parquet"))
    if len(paths) < 2: raise FileNotFoundError("Dense teacher requires per-sample results from at least two bridges")
    merged = None; names = []
    keys = ["sample_key", "task_label"]
    for path in paths:
        name = path.parent.name; names.append(name)
        frame = pd.read_parquet(path)
        frame = frame[frame["candidate_depth"].eq(depth)].copy()
        retrieval = frame[frame["direction"].notna()].copy()
        retrieval["sample_key"] = "retrieval:" + retrieval["direction"].astype(str) + ":" + retrieval["query_index"].astype(str)
        retrieval["task_label"] = retrieval["direction"].astype(str)
        retrieval[name] = 1.0 / retrieval["positive_rank"].astype(float).clip(lower=1.0)
        classification = frame[frame["direction"].isna() & frame["task"].notna()].copy()
        identifier = classification["sample_id"].astype(str) if "sample_id" in classification else classification["query_index"].astype(str)
        classification["sample_key"] = "classification:" + classification["task"].astype(str) + ":" + identifier
        classification["task_label"] = classification["task"].astype(str)
        classification[name] = classification["correct"].astype(float)
        frame = pd.concat([retrieval[keys + [name]], classification[keys + [name]]], ignore_index=True)
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner")
    if merged is None or merged.empty: raise RuntimeError("No aligned bridge rows exist for dense-teacher training")
    scores = torch.tensor(merged[names].to_numpy(), dtype=torch.float32)
    tasks, task_names = _task_codes(merged["task_label"])
    targets = scores.max(dim=1).values
    return scores, tasks, targets, names, task_names


def _teacher_training_data(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    paths = sorted((root / "results/phase2/reranking/runs").glob("*/training_pair_scores.parquet"))
    if len(paths) < 2: raise FileNotFoundError("Dense teacher requires aligned training-split scores from at least two bridges")
    merged = None; names = []; keys = ["sample_key", "task_label"]
    for path in paths:
        name = path.parent.name; names.append(name); frame = pd.read_parquet(path)
        if not frame["allowed_for_training"].astype(bool).all(): raise RuntimeError(f"Non-training rows found in {path}")
        frame = frame[keys + ["positive_score", "negative_score"]].rename(columns={"positive_score": f"{name}__positive", "negative_score": f"{name}__negative"})
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner")
    if merged is None or merged.empty: raise RuntimeError("No aligned training rows exist for dense teacher")
    positive = torch.tensor(merged[[f"{name}__positive" for name in names]].to_numpy(), dtype=torch.float32)
    negative = torch.tensor(merged[[f"{name}__negative" for name in names]].to_numpy(), dtype=torch.float32)
    task_values, task_names = _task_codes(merged["task_label"])
    scores = torch.cat([positive, negative]); tasks = torch.cat([task_values, task_values]); targets = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
    return scores, tasks, targets, names, task_names


def train_dense_teacher(
    project_root: str | Path,
    config: dict,
    resume: bool = True,
    cleanup_process_group: bool = True,
) -> dict:
    root = Path(project_root).resolve(); context = initialise_distributed(); seed_rank(int(config.get("seed", 42)), context.rank)
    depth = int(config.get("candidate_depth", 64)); scores, tasks, targets, names, task_names = _teacher_training_data(root)
    expected_tasks = sorted(map(str, config.get("task_vocabulary", task_names)))
    if task_names != expected_tasks:
        raise ValueError(f"Dense training task vocabulary is {task_names}, expected {expected_tasks}")
    dataset = TensorDataset(scores, tasks, targets); sampler = sampler_for(dataset, context, True)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1024)), sampler=sampler, shuffle=sampler is None)
    model = DenseTaskConditionedTeacher(len(names), int(tasks.max()) + 1, strategy=str(config.get("strategy", "task_and_input")), hidden_dim=int(config.get("hidden_dim", 64)), path_dropout=float(config.get("path_dropout", 0.05))).to(context.device)
    wrapped = _wrap_dense_teacher(model, context)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, int(config.get("epochs", 20))))
    amp_enabled = bool(config.get("amp", True)) and context.device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    strategy = str(config.get("strategy", "task_and_input")); run_name = str(config.get("run_name", strategy)); run_dir = root / "results/phase2/dense_teacher/runs" / run_name; checkpoints = root / "results/phase2/dense_teacher/checkpoints" / run_name; checkpoints.mkdir(parents=True, exist_ok=True)
    last, best = checkpoints / "last.pt", checkpoints / "best.pt"; cfg_fp = fingerprint(config); start, best_loss = 0, math.inf
    data_fp = fingerprint({"source": "phase2-training-pair-scores", "paths": names, "tasks": task_names})
    if resume and last.exists():
        metadata = torch.load(last, map_location="cpu", weights_only=False)
        compatible = metadata.get("config_fingerprint") == cfg_fp and metadata.get("dataset_fingerprint") == data_fp and metadata.get("task_names") == task_names
        if compatible:
            state = load_checkpoint(last, wrapped, optimizer, scheduler, scaler, context.device); start = int(state["epoch"]) + 1; best_loss = float(state["best_loss"])
    install_preemption_handlers(); metrics = []
    try:
        manager = managed_stage(run_dir, "dense_teacher_training", config, root) if rank_zero(context) else _null_context(run_dir)
        with manager:
            for epoch in range(start, int(config.get("epochs", 20))):
                if sampler is not None: sampler.set_epoch(epoch)
                total = torch.zeros((), device=context.device); batches = 0
                for path_scores, task_ids, target_values in loader:
                    path_scores, task_ids, target_values = path_scores.to(context.device), task_ids.to(context.device), target_values.to(context.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(context.device.type, enabled=amp_enabled):
                        output = wrapped(path_scores, task_ids); loss = torch.nn.functional.binary_cross_entropy_with_logits(output["score"], target_values) + teacher_regularisation(output["weights"], float(config.get("entropy_weight", 0.001)), float(config.get("balance_weight", 0.01)))
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += loss.detach(); batches += 1
                    if PreemptionState.requested: break
                epoch_loss = float(reduce_mean(total / max(1, batches), context)); scheduler.step()
                if rank_zero(context):
                    metrics.append({"epoch": epoch, "loss": epoch_loss}); atomic_csv(pd.DataFrame(metrics), run_dir / "metrics_by_epoch.csv")
                    state = checkpoint_payload(wrapped, optimizer, scheduler, scaler, epoch=epoch, global_step=(epoch + 1) * len(loader), best_loss=min(best_loss, epoch_loss), config_fingerprint=cfg_fp, dataset_fingerprint=data_fp, world_size=context.world_size, path_names=names, task_names=task_names)
                    atomic_torch_save(state, last)
                    if epoch_loss < best_loss: best_loss = epoch_loss; state["best_loss"] = best_loss; atomic_torch_save(state, best)
                if PreemptionState.requested:
                    if rank_zero(context): write_status(run_dir, RunStatus.INTERRUPTED, stage="dense_teacher_training")
                    raise KeyboardInterrupt("pre-termination checkpoint completed")
            if rank_zero(context):
                summary = {"status": "COMPLETED", "strategy": strategy, "best_loss": best_loss, "paths": names, "task_names": task_names, "world_size": context.world_size, "best_checkpoint": str(best), "config_fingerprint": cfg_fp, "dataset_fingerprint": data_fp, "training_source": "phase2_training_hard_negatives", "protected_evaluation_rows_used_for_training": 0}
                complete_stage(run_dir, "dense_teacher_training", summary)
            else: summary = {"status": "COMPLETED", "rank": context.rank}
    finally:
        if cleanup_process_group:
            cleanup_distributed()
    return summary


@torch.no_grad()
def evaluate_dense_checkpoint(project_root: str | Path, config: dict, checkpoint: str | Path) -> dict[str, float]:
    root = Path(project_root).resolve(); scores, tasks, _, names, task_names = _teacher_evaluation_data(root, int(config.get("candidate_depth", 64)))
    strategy = str(config.get("strategy", "task_and_input")); state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_tasks = list(map(str, state.get("task_names", [])))
    if checkpoint_tasks != task_names:
        raise ValueError(f"Dense checkpoint task vocabulary is {checkpoint_tasks}, evaluation requires {task_names}")
    model = DenseTaskConditionedTeacher(len(names), len(task_names), strategy=strategy, hidden_dim=int(config.get("hidden_dim", 64)), path_dropout=0.0)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state["model"].items()}); model.eval()
    started = time.perf_counter(); output = model(scores, tasks); elapsed = time.perf_counter() - started
    return {
        "mean_sample_utility": float(output["score"].mean()),
        "latency_ms": elapsed * 1000.0 / max(1, len(scores)),
        "peak_memory_bytes": 0.0,
        "gating_entropy": float(output["gating_entropy"].mean()),
        "path_utilisation_variance": float(output["weights"].mean(0).var()),
    }


class _null_context:
    def __init__(self, value): self.value = value
    def __enter__(self): return self.value
    def __exit__(self, *args): return False


@torch.no_grad()
def evaluate_dense_teacher(project_root: str | Path, config: dict) -> dict:
    root = Path(project_root).resolve(); depth = int(config.get("candidate_depth", 64)); scores, tasks, targets, names, task_names = _teacher_evaluation_data(root, depth)
    destination = root / "results/phase2/dense_teacher"; destination.mkdir(parents=True, exist_ok=True)
    result_rows, weight_frames = [], []
    training_scores, _, training_targets, training_names, training_task_names = _teacher_training_data(root)
    if training_names != names: raise RuntimeError("Dense training and evaluation path order differs")
    if training_task_names != task_names: raise RuntimeError(f"Dense training tasks {training_task_names} differ from evaluation tasks {task_names}")
    best_path = int(((training_scores > 0).eq(training_targets.unsqueeze(1).bool()).float().mean(0)).argmax())
    baseline_outputs = {
        "best_single": scores[:, best_path],
        "uniform": scores.mean(1),
    }
    for method, values in baseline_outputs.items():
        result_rows.append({"method": method, "task": "multitask", "metric": "mean_sample_utility", "value": float(values.mean()), "candidate_depth": depth})
    strategies = config.get("strategies", [config.get("strategy", "task_and_input")])
    for strategy in strategies:
        state = torch.load(destination / "checkpoints" / strategy / "best.pt", map_location="cpu", weights_only=False)
        checkpoint_tasks = list(map(str, state.get("task_names", [])))
        if checkpoint_tasks != task_names: raise ValueError(f"{strategy} checkpoint tasks {checkpoint_tasks} differ from evaluation tasks {task_names}")
        model = DenseTaskConditionedTeacher(len(names), len(task_names), strategy=strategy, hidden_dim=int(config.get("hidden_dim", 64)), path_dropout=0.0)
        model.load_state_dict({key.removeprefix("module."): value for key, value in state["model"].items()}); model.eval(); output = model(scores, tasks)
        result_rows.append({"method": strategy, "task": "multitask", "metric": "mean_sample_utility", "value": float(output["score"].mean()), "candidate_depth": depth})
        frame = pd.DataFrame(output["weights"].numpy(), columns=names); frame["method"] = strategy; frame["task_id"] = tasks.numpy(); frame["gating_entropy"] = output["gating_entropy"].numpy(); frame["dominant_path"] = output["dominant_path"].numpy(); weight_frames.append(frame)
    result = pd.DataFrame(result_rows); atomic_csv(result, destination / "results_best.csv")
    weights = pd.concat(weight_frames, ignore_index=True); weight_target = destination / "per_sample_weights.parquet"; weight_temporary = destination / "per_sample_weights.parquet.tmp"; weights.to_parquet(weight_temporary, index=False); os.replace(weight_temporary, weight_target)
    stats = weights.groupby("method")[names + ["gating_entropy"]].agg(["mean", "var"]); stats.columns = ["__".join(value) for value in stats.columns]; atomic_csv(stats.reset_index(), destination / "path_statistics.csv")
    canonical = str(config.get("strategy", "task_and_input")); run = destination / "runs" / canonical; atomic_csv(result, run / "results.csv")
    atomic_text("# Dense Teacher\n\nAll selected paths execute. Eight combination strategies are trained only on Phase 2 training-split hard negatives and evaluated on held-out COCO validation/classification-development evidence. Learned path weights are normalised to one.\n", destination / "summary.md")
    return {"status": "COMPLETED", "rows": len(weights), "paths": names, "strategies": ["best_single", "uniform", *strategies], "training_source": "phase2_training_hard_negatives", "protected_evaluation_rows_used_for_training": 0}
