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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.training.train import train_from_config
from src.utils.checkpoint import load_checkpoint_metrics
from src.utils.config import deep_update, load_config


VARIANT_FLAGS = {
    "baseline": (False, False),
    "local": (True, False),
    "global": (False, True),
    "local_global": (True, True),
}

FIELDNAMES = [
    "comparison",
    "run_name",
    "vision_encoder",
    "text_encoder",
    "variant",
    "use_local_blf",
    "use_global_blf",
    "seed",
    "batch_size",
    "epochs_requested",
    "selection_metric",
    "best_epoch",
    "completed_epoch",
    "stopped_early",
    "val_loss",
    "i2t_R@1",
    "i2t_R@5",
    "i2t_R@10",
    "t2i_R@1",
    "t2i_R@5",
    "t2i_R@10",
    "mean_rank_i2t",
    "median_rank_i2t",
    "mean_rank_t2i",
    "median_rank_t2i",
    "num_image_queries",
    "num_text_queries",
    "params_total",
    "params_trainable",
    "coco5_val_loss",
    "coco5_i2t_R@1",
    "coco5_i2t_R@5",
    "coco5_i2t_R@10",
    "coco5_t2i_R@1",
    "coco5_t2i_R@5",
    "coco5_t2i_R@10",
    "coco5_mean_rank_i2t",
    "coco5_median_rank_i2t",
    "coco5_mean_rank_t2i",
    "coco5_median_rank_t2i",
    "coco5_num_image_queries",
    "coco5_num_text_queries",
    "best_checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible baseline/BLF seed comparisons across one or more GPUs.")
    parser.add_argument("--sweep_config", default="configs/seed_sweep.yaml")
    parser.add_argument("--gpus", nargs="+", default=None, help="Visible GPU IDs, for example --gpus 0 1 2 3.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_job", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_result", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_gpus(gpus: list[str] | None) -> list[str]:
    if not gpus:
        return []
    return [gpu.strip() for item in gpus for gpu in item.split(",") if gpu.strip()]


def resolve_gpu_tokens(gpus: list[str]) -> dict[str, str]:
    inherited = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    resolved: dict[str, str] = {}
    for gpu in gpus:
        if inherited and gpu.isdigit() and int(gpu) < len(inherited):
            resolved[gpu] = inherited[int(gpu)]
        else:
            resolved[gpu] = gpu
    return resolved


def load_sweep(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sweep_path = Path(path)
    if not sweep_path.is_absolute():
        sweep_path = PROJECT_ROOT / sweep_path
    sweep = load_config(sweep_path)
    base_config_path = Path(str(sweep["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    base_config = load_config(base_config_path)
    base_config = deep_update(base_config, dict(sweep.get("config_overrides", {})))
    return sweep, base_config


def build_jobs(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in sweep.get("seeds", [42, 43, 44, 45, 46])]
    checkpoint_root = Path(str(sweep.get("checkpoint_root", "checkpoints/seed_sweep")))
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for comparison in sweep.get("comparisons", []):
        name = str(comparison["name"])
        vision_encoder = str(comparison["vision_encoder"])
        text_encoder = str(comparison["text_encoder"])
        for variant in comparison.get("variants", ["baseline", "local_global"]):
            variant = str(variant)
            if variant not in VARIANT_FLAGS:
                raise ValueError(f"Unknown variant {variant!r} in comparison {name!r}")
            for seed in seeds:
                run_name = f"{name}_{variant}_seed{seed}"
                if run_name in seen:
                    raise ValueError(f"Duplicate sweep run name: {run_name}")
                seen.add(run_name)
                jobs.append(
                    {
                        "comparison": name,
                        "run_name": run_name,
                        "vision_encoder": vision_encoder,
                        "text_encoder": text_encoder,
                        "variant": variant,
                        "seed": seed,
                        "save_dir": (checkpoint_root / run_name).as_posix(),
                    }
                )
    if not jobs:
        raise ValueError("Sweep config did not define any jobs")
    return jobs


def read_completed_runs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {str(row.get("run_name")) for row in csv.DictReader(handle) if row.get("run_name")}


def append_unique_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    existing = read_completed_runs(path)
    new_rows = [row for row in rows if str(row["run_name"]) not in existing]
    if not new_rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def job_config(base_config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base_config)
    use_local, use_global = VARIANT_FLAGS[str(job["variant"])]
    config["seed"] = int(job["seed"])
    config["model"]["vision_encoder"] = str(job["vision_encoder"])
    config["model"]["text_encoder"] = str(job["text_encoder"])
    config["model"]["use_local_blf"] = use_local
    config["model"]["use_global_blf"] = use_global
    config["training"]["save_dir"] = str(job["save_dir"])
    return config


def evaluate_coco5(sweep: dict[str, Any], training_config: dict[str, Any], best_path: Path) -> dict[str, float]:
    evaluation_config_path = sweep.get("evaluation_config")
    if not evaluation_config_path:
        return {}
    from src.data import build_dataloaders
    from src.training.evaluate import evaluate_model
    from src.training.train import build_model_from_config
    from src.utils.checkpoint import load_checkpoint
    from src.utils.device import get_device

    path = Path(str(evaluation_config_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = load_config(path)
    config["model"] = deepcopy(training_config["model"])
    device = get_device(str(config.get("device", "auto")))
    _, val_loader = build_dataloaders(config)
    model = build_model_from_config(config).to(device)
    load_checkpoint(best_path, model, map_location=device)
    metrics = evaluate_model(model, val_loader, device, k_values=list(config.get("evaluation", {}).get("k_values", [1, 5, 10])))
    return {f"coco5_{key}": value for key, value in metrics.items()}


def run_worker(args: argparse.Namespace, sweep: dict[str, Any], base_config: dict[str, Any]) -> None:
    if args.worker_result is None:
        raise ValueError("--worker_result is required with --worker_job")
    job = json.loads(args.worker_job)
    config = job_config(base_config, job)
    save_dir = PROJECT_ROOT / str(job["save_dir"])
    best_path = save_dir / "best.pt"
    final_path = save_dir / "final.pt"
    if args.resume and final_path.exists() and best_path.exists():
        metrics = load_checkpoint_metrics(best_path)
    else:
        metrics = train_from_config(config, resume=args.resume)
    summary_path = save_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    coco5_metrics = evaluate_coco5(sweep, config, best_path)
    use_local, use_global = VARIANT_FLAGS[str(job["variant"])]
    row = {
        **job,
        "use_local_blf": use_local,
        "use_global_blf": use_global,
        "batch_size": config["training"].get("batch_size"),
        "epochs_requested": config["training"].get("epochs"),
        "selection_metric": summary.get("selection_metric", "i2t_R@1"),
        "best_epoch": summary.get("best_epoch", "unknown"),
        "completed_epoch": summary.get("completed_epoch", "unknown"),
        "stopped_early": summary.get("stopped_early", "unknown"),
        **metrics,
        **coco5_metrics,
        "best_checkpoint": best_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    result_path = PROJECT_ROOT / args.worker_result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def run_parallel(args: argparse.Namespace, jobs: list[dict[str, Any]], results_path: Path) -> None:
    gpus = normalize_gpus(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU must be supplied with --gpus")
    gpu_tokens = resolve_gpu_tokens(gpus)
    run_id = f"{int(time.time())}_{os.getpid()}"
    shard_dir = PROJECT_ROOT / "results" / ".seed_sweep_shards" / run_id
    log_dir = PROJECT_ROOT / "logs" / "seed_sweep" / run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(jobs, start=1))
    available_gpus = list(gpus)
    active: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    print(f"Running {len(jobs)} jobs across {len(gpus)} GPUs: {', '.join(gpus)}")
    print(f"Per-job logs: {log_dir}")
    try:
        while pending or active:
            while pending and available_gpus:
                index, job = pending.pop(0)
                gpu = available_gpus.pop(0)
                shard_path = shard_dir / f"job_{index:03d}_{job['run_name']}.csv"
                log_path = log_dir / f"job_{index:03d}_{job['run_name']}.log"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--sweep_config",
                    args.sweep_config,
                    "--worker_job",
                    json.dumps(job, separators=(",", ":")),
                    "--worker_result",
                    shard_path.relative_to(PROJECT_ROOT).as_posix(),
                ]
                if args.resume:
                    command.append("--resume")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_tokens[gpu]
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                log_handle = log_path.open("w")
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active.append({"process": process, "handle": log_handle, "gpu": gpu, "job": job, "shard": shard_path, "log": log_path})
                print(f"[GPU {gpu}] started {index}/{len(jobs)}: {job['run_name']}")
            time.sleep(5)
            for item in active[:]:
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].close()
                active.remove(item)
                available_gpus.append(str(item["gpu"]))
                if returncode != 0:
                    raise RuntimeError(f"Job failed: {item['job']['run_name']}. See {item['log']}")
                with Path(item["shard"]).open(newline="") as handle:
                    completed_rows.extend(csv.DictReader(handle))
                print(f"[GPU {item['gpu']}] finished: {item['job']['run_name']}")
    except Exception:
        for item in active:
            item["process"].terminate()
            item["handle"].close()
        raise
    added = append_unique_rows(results_path, completed_rows)
    print(f"Added {added} rows to {results_path}")


def main() -> None:
    args = parse_args()
    sweep, base_config = load_sweep(args.sweep_config)
    if args.worker_job is not None:
        run_worker(args, sweep, base_config)
        return
    results_path = PROJECT_ROOT / str(sweep.get("results_path", "results/seed_sweep_results.csv"))
    completed = read_completed_runs(results_path)
    jobs = []
    for job in build_jobs(sweep):
        final_exists = (PROJECT_ROOT / str(job["save_dir"]) / "final.pt").exists()
        if args.resume and job["run_name"] in completed and final_exists:
            print(f"Skipping completed run: {job['run_name']}")
            continue
        jobs.append(job)
    if not jobs:
        print("No pending seed-sweep jobs.")
        return
    run_parallel(args, jobs, results_path)


if __name__ == "__main__":
    main()
