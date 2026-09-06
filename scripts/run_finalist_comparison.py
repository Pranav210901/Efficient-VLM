from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.training.train import train_from_config
from src.utils.config import load_config
from src.utils.gpu_jobs import normalize_gpu_ids, resolve_gpu_tokens


JOBS = (
    {
        "run_name": "baseline_convnext_tiny_minilm_l6",
        "headline": "Baseline ConvNeXt-Tiny + MiniLM",
        "variant": "baseline",
        "use_local_blf": False,
        "use_global_blf": False,
        "save_dir": "checkpoints/finalist_convnext_tiny_minilm_l6_baseline_15ep",
    },
    {
        "run_name": "blf_local_global_convnext_tiny_minilm_l6",
        "headline": "BLF ConvNeXt-Tiny + MiniLM",
        "variant": "local_global",
        "use_local_blf": True,
        "use_global_blf": True,
        "save_dir": "checkpoints/finalist_convnext_tiny_minilm_l6_local_global_15ep",
    },
)


def normalize_gpus(values: list[str] | None) -> list[str]:
    return normalize_gpu_ids(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two historical finalist jobs with one independent process per GPU.")
    parser.add_argument("--config", default="configs/coco_blf_rtxpro.yaml")
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--results_path", default="results/finalist_convnext_tiny_minilm_l6_15ep.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_job", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_result", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_config(base: dict[str, Any], job: dict[str, Any], epochs: int) -> dict[str, Any]:
    config = deepcopy(base)
    config.setdefault("model", {}).update(
        {
            "vision_encoder": "convnext_tiny",
            "text_encoder": "minilm_l6",
            "use_local_blf": bool(job["use_local_blf"]),
            "use_global_blf": bool(job["use_global_blf"]),
        }
    )
    config.setdefault("training", {}).update({"epochs": epochs, "save_dir": str(job["save_dir"])})
    return config


def summarize(job: dict[str, Any], config: dict[str, Any], epochs: int) -> dict[str, Any]:
    best_path = PROJECT_ROOT / str(job["save_dir"]) / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint after finalist job: {best_path}")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {})
    i2t = float(metrics.get("i2t_R@1", 0.0)); t2i = float(metrics.get("t2i_R@1", 0.0))
    return {
        "headline": job["headline"], "run_name": job["run_name"], "vision_encoder": "convnext_tiny",
        "text_encoder": "minilm_l6", "variant": job["variant"], "use_local_blf": job["use_local_blf"],
        "use_global_blf": job["use_global_blf"], "epochs_requested": epochs, "seed": int(config.get("seed", 42)),
        "selection_metric": "i2t_R@1", "best_epoch": int(checkpoint.get("epoch", 0)), "best_i2t_R@1": i2t,
        "best_t2i_R@1": t2i, "best_mean_R@1": (i2t + t2i) / 2,
        "best_i2t_R@5": metrics.get("i2t_R@5"), "best_i2t_R@10": metrics.get("i2t_R@10"),
        "best_t2i_R@5": metrics.get("t2i_R@5"), "best_t2i_R@10": metrics.get("t2i_R@10"),
        "best_val_loss": metrics.get("val_loss"), "mean_rank_i2t": metrics.get("mean_rank_i2t"),
        "mean_rank_t2i": metrics.get("mean_rank_t2i"), "params_total": metrics.get("params_total"),
        "params_trainable": metrics.get("params_trainable"), "best_checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
    }


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def run_worker(args: argparse.Namespace) -> None:
    if not args.worker_result:
        raise ValueError("--worker_result is required with --worker_job")
    job = json.loads(args.worker_job)
    config = build_config(load_config(PROJECT_ROOT / args.config), job, args.epochs)
    final_path = PROJECT_ROOT / str(job["save_dir"]) / "final.pt"
    if not (args.resume and final_path.exists()):
        train_from_config(config, resume=args.resume)
    write_row(PROJECT_ROOT / args.worker_result, summarize(job, config, args.epochs))


def run_parallel(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[Path]:
    gpus = normalize_gpus(args.gpus)
    if not gpus:
        raise ValueError("At least one visible GPU is required")
    gpu_tokens = resolve_gpu_tokens(gpus)
    run_id = f"{int(time.time())}_{os.getpid()}"
    shard_dir = PROJECT_ROOT / "results/.finalist_shards" / run_id
    log_dir = PROJECT_ROOT / "logs/finalist" / run_id
    shard_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(jobs, 1)); available = list(gpus); active: list[dict[str, Any]] = []; completed: list[Path] = []
    print(f"Running {len(jobs)} finalist jobs across {min(len(jobs), len(gpus))} of {len(gpus)} visible GPUs")
    while pending or active:
        while pending and available:
            index, job = pending.pop(0); gpu = available.pop(0)
            shard = shard_dir / f"job_{index:02d}_{job['variant']}.csv"; log = log_dir / f"job_{index:02d}_{job['variant']}.log"
            command = [sys.executable, str(Path(__file__).resolve()), "--config", args.config, "--epochs", str(args.epochs), "--worker_job", json.dumps(job), "--worker_result", str(shard.relative_to(PROJECT_ROOT))]
            if args.resume: command.append("--resume")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu_tokens[gpu]; env["PYTHONUNBUFFERED"] = "1"
            handle = log.open("w"); process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append({"process": process, "handle": handle, "gpu": gpu, "job": job, "shard": shard, "log": log})
            print(f"[GPU {gpu}] started {job['run_name']}")
        time.sleep(2)
        for item in active[:]:
            code = item["process"].poll()
            if code is None: continue
            item["handle"].close(); active.remove(item); available.append(item["gpu"])
            if code != 0:
                for other in active: other["process"].terminate(); other["handle"].close()
                raise RuntimeError(f"Finalist job failed: {item['job']['run_name']}. See {item['log']}")
            completed.append(item["shard"]); print(f"[GPU {item['gpu']}] finished {item['job']['run_name']}")
    return completed


def merge(shards: list[Path], output: Path) -> None:
    rows = []
    for shard in shards:
        with shard.open(newline="") as handle: rows.extend(csv.DictReader(handle))
    if not rows: raise RuntimeError("No finalist result rows were produced")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} finalist rows to {output}")


def main() -> None:
    args = parse_args()
    if args.worker_job:
        run_worker(args); return
    shards = run_parallel(args, list(JOBS))
    merge(shards, PROJECT_ROOT / args.results_path)


if __name__ == "__main__":
    main()
