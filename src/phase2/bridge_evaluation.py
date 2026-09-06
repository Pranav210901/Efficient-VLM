from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.phase15.io_utils import atomic_csv
from .bridge_model import FrozenPairBridge
from .checkpointing import load_checkpoint
from .evaluator import evaluate_classification_reranking, evaluate_coco_reranking
from .datasets import ClassificationHardNegativeDataset, RetrievalHardNegativeDataset, bridge_collate
from .hard_negative_loader import read_hard_negatives
from .prerequisites import load_locked_selection, primary_paths
from src.data.transforms import build_image_transform
from src.phase15.training_hard_negative_mining import _classification_training_dataset


def _limit_classification_per_task(frame: pd.DataFrame, rows_per_task: int) -> pd.DataFrame:
    if rows_per_task < 1:
        raise ValueError("dense-teacher classification rows per task must be positive")
    return (
        frame.groupby("source_task", sort=False, group_keys=False)
        .head(rows_per_task)
        .reset_index(drop=True)
    )


def evaluate_bridge_checkpoint(root: Path, pair: dict, config: dict, checkpoint: Path, run_dir: Path) -> dict:
    pair_id = str(pair["pair_id"]); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model_cfg = config.get("model", config)
    model = FrozenPairBridge.from_selected_pair(pair, device, bridge_dim=int(model_cfg.get("bridge_dim", 256)), attention_heads=int(model_cfg.get("attention_heads", 4)), layers=int(model_cfg.get("layers", 1)), feedforward_dim=int(model_cfg.get("feedforward_dim", 512)), dropout=float(model_cfg.get("dropout", 0.1)), pooling=str(model_cfg.get("pooling", "masked_mean")))
    load_checkpoint(checkpoint, model, map_location=device); model.eval()
    phase1 = torch.load(Path(pair["checkpoint_path"]), map_location="cpu", weights_only=False)["config"]
    depths = config.get("candidate_depths", [16, 32, 64, 128]); image_size = int(phase1["data"].get("image_size", 224)); token_batch = int(config.get("token_batch_size", 64)); pair_batch = int(config.get("pair_batch_size", 128))
    dual_score_weight = float(config.get("dual_score_weight", 0.5))
    cache_path = root / "results/phase1_multitask/cache" / f"{pair_id}__coco__full.pt"; cached = torch.load(cache_path, map_location="cpu", weights_only=False)["payload"]
    coco_csv = root / str(config.get("coco_csv", "data/val_all_captions.csv"))
    results, samples, coverage = evaluate_coco_reranking(model, cached, coco_csv, root, depths, image_size, token_batch, pair_batch, device, dual_score_weight=dual_score_weight); results["task"] = "coco_retrieval"
    classification_results, classification_samples, classification_coverage = [], [], []
    for task in config.get("tasks", ["coco_retrieval"])[1:]:
        cache = root / "results/phase1_multitask_development/cache" / f"{pair_id}__{task}__development__full.pt"; payload = torch.load(cache, map_location="cpu", weights_only=False)["payload"]
        values = evaluate_classification_reranking(model, payload, task, root, depths, image_size, token_batch, pair_batch, device, dual_score_weight=dual_score_weight)
        classification_results.append(values[0]); classification_samples.append(values[1]); classification_coverage.append(values[2])
    if classification_results:
        results = pd.concat([results, *classification_results], ignore_index=True); samples = pd.concat([samples, *classification_samples], ignore_index=True); coverage = pd.concat([coverage, *classification_coverage], ignore_index=True)
    results["method"] = run_dir.name; results["split_role"] = "development"; samples["method"] = run_dir.name; coverage["method"] = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True); atomic_csv(results, run_dir / "results.csv"); atomic_csv(coverage, run_dir / "candidate_coverage.csv")
    atomic_csv(results[["task", "candidate_depth", "metric", "value"]], run_dir / "metrics_by_epoch.csv")
    sample_target = run_dir / "per_sample_results.parquet"; sample_temporary = run_dir / "per_sample_results.parquet.tmp"; samples.to_parquet(sample_temporary, index=False); os.replace(sample_temporary, sample_target)
    latency = results[results["metric"].isin(["reranking_latency_ms", "additional_latency_ms"])].copy().rename(columns={"value": "latency_ms"}); performance = results[results["metric"].isin(["recall_at_1", "top1"])][["task", "candidate_depth", "value"]].rename(columns={"value": "performance"})
    efficiency = latency.merge(performance, on=["task", "candidate_depth"], how="left"); efficiency["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0; efficiency["throughput_per_second"] = 1000.0 / efficiency["latency_ms"].clip(lower=1e-9); efficiency["offline_index_cost"] = "reused_phase1_dual_encoder_cache"; efficiency["online_query_cost"] = efficiency["candidate_depth"].astype(str) + " cross-attention pairs"; atomic_csv(efficiency, run_dir / "efficiency.csv")
    training_rows = score_dense_training_pairs(root, model, config, image_size, run_dir, device) if config.get("write_dense_training_scores", False) else 0
    return {"result_rows": len(results), "sample_rows": len(samples), "dense_training_rows": training_rows, "checkpoint": str(checkpoint)}


@torch.no_grad()
def score_dense_training_pairs(root: Path, model, config: dict, image_size: int, run_dir: Path, device: torch.device) -> int:
    canonical = primary_paths(load_locked_selection(root))[0]["pair_id"]
    maximum = int(config.get("dense_teacher_training_samples_per_source", 20000)); data_root = root / "results/phase2/data"
    classification_per_task = int(config.get("dense_teacher_classification_samples_per_task", maximum))
    retrieval = read_hard_negatives(data_root / "retrieval_train.parquet", kind="retrieval", config_id=canonical, max_rows=maximum)
    classification = read_hard_negatives(data_root / "classification_train.parquet", kind="classification", config_id=canonical)
    classification = _limit_classification_per_task(classification, classification_per_task)
    transform = build_image_transform(image_size, train=False)
    task_datasets = {}
    for task in classification["source_task"].unique(): task_datasets[str(task)] = _classification_training_dataset(root, str(task), image_size, "full")[0]
    def resolve(row): return task_datasets[str(row.source_task)][int(row.sample_index)][0]
    dataset = ConcatDataset([RetrievalHardNegativeDataset(retrieval, transform), ClassificationHardNegativeDataset(classification, resolve, transform)])
    loader = DataLoader(dataset, batch_size=int(config.get("token_batch_size", 64)), shuffle=False, num_workers=4, pin_memory=device.type == "cuda", collate_fn=bridge_collate)
    positive_parts, negative_parts = [], []
    for batch in loader:
        positive_parts.append(model(batch["positive_images"].to(device), batch["positive_captions"])["scores"].float().cpu())
        negative_parts.append(model(batch["negative_images"].to(device), batch["negative_captions"])["scores"].float().cpu())
    retrieval_keys = "r:" + retrieval["direction"].astype(str) + ":" + retrieval["query_id"].astype(str) + ":" + retrieval["negative_id"].astype(str)
    classification_keys = "c:" + classification["source_task"].astype(str) + ":" + classification["sample_id"].astype(str) + ":" + classification["negative_class"].astype(str)
    frame = pd.DataFrame({
        "sample_key": pd.concat([retrieval_keys, classification_keys], ignore_index=True),
        "task_label": pd.concat([retrieval["direction"].astype(str), classification["source_task"].astype(str)], ignore_index=True),
        "positive_score": torch.cat(positive_parts).numpy(), "negative_score": torch.cat(negative_parts).numpy(),
        "source_split": pd.concat([retrieval["source_split"], classification["source_split"]], ignore_index=True),
        "allowed_for_training": True,
    })
    target = run_dir / "training_pair_scores.parquet"; temporary = run_dir / "training_pair_scores.parquet.tmp"; frame.to_parquet(temporary, index=False); os.replace(temporary, target)
    return len(frame)
