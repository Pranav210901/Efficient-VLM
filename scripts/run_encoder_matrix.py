from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.training.train import train_from_config
from src.utils.checkpoint import load_checkpoint_metrics
from src.utils.config import load_config
from src.utils.gpu_jobs import normalize_gpu_ids, resolve_gpu_tokens


DEFAULT_VISION_ENCODERS = ["efficientnet_b0", "convnext_tiny"]
DEFAULT_TEXT_ENCODERS = ["minilm_l6", "distilbert"]

BASELINE = [(False, False)]
BLF_VARIANTS = [(True, False), (False, True), (True, True)]

FIELDNAMES = [
    "vision_encoder",
    "text_encoder",
    "vision_setup",
    "fusion_type",
    "use_local_blf",
    "use_global_blf",
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
    "params_total",
    "params_trainable",
]

VARIANT_FLAGS = {
    "baseline": (False, False),
    "local": (True, False),
    "global": (False, True),
    "local_global": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VE/TE pairs and BLF variants for an alignment matrix.")
    parser.add_argument("--config", default="configs/experiment_blf.yaml")
    parser.add_argument("--vision_encoders", nargs="+", default=DEFAULT_VISION_ENCODERS)
    parser.add_argument("--text_encoders", nargs="+", default=DEFAULT_TEXT_ENCODERS)
    parser.add_argument("--mode", choices=["baseline", "blf", "all"], default="all")
    parser.add_argument("--fusion_type", default=None)
    parser.add_argument("--results_path", default="results/alignment_matrix_results.csv")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU IDs to use for parallel matrix jobs, e.g. --gpus 0 1 2 3 4 or --gpus 0,1,2,3,4.",
    )
    parser.add_argument(
        "--matrix_variant",
        choices=list(VARIANT_FLAGS),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished jobs from checkpoints and skip jobs that already have final checkpoints.",
    )
    return parser.parse_args()


def normalize_gpus(gpus: list[str] | None) -> list[str]:
    return normalize_gpu_ids(gpus)


def setup_name(vision_encoder: str, use_local: bool, use_global: bool) -> str:
    if use_local and use_global:
        suffix = "local+global"
    elif use_local:
        suffix = "local"
    elif use_global:
        suffix = "global"
    else:
        suffix = "only"
    return f"{vision_encoder} {suffix}"


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def result_row_exists(path: Path, vision_encoder: str, text_encoder: str, use_local: bool, use_global: bool) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row.get("vision_encoder") == vision_encoder
                and row.get("text_encoder") == text_encoder
                and row.get("use_local_blf") == str(use_local)
                and row.get("use_global_blf") == str(use_global)
            ):
                return True
    return False


def variants_for_mode(mode: str) -> list[tuple[bool, bool]]:
    if mode == "baseline":
        return BASELINE
    if mode == "blf":
        return BLF_VARIANTS
    return BASELINE + BLF_VARIANTS


def variant_name(use_local: bool, use_global: bool) -> str:
    if use_local and use_global:
        return "local_global"
    if use_local:
        return "local"
    if use_global:
        return "global"
    return "baseline"


def variants_for_args(args: argparse.Namespace) -> list[tuple[bool, bool]]:
    if args.matrix_variant:
        return [VARIANT_FLAGS[args.matrix_variant]]
    return variants_for_mode(args.mode)


def iter_jobs(args: argparse.Namespace) -> list[tuple[str, str, bool, bool]]:
    jobs = [
        (vision_encoder, text_encoder, use_local, use_global)
        for use_local, use_global in variants_for_args(args)
        for vision_encoder in args.vision_encoders
        for text_encoder in args.text_encoders
    ]
    return dedupe_jobs(jobs)


def dedupe_jobs(jobs: list[tuple[str, str, bool, bool]]) -> list[tuple[str, str, bool, bool]]:
    deduped: list[tuple[str, str, bool, bool]] = []
    seen: set[tuple[str, str, bool, bool]] = set()
    for job in jobs:
        if job in seen:
            vision_encoder, text_encoder, use_local, use_global = job
            print(f"Skipping duplicate requested job: {vision_encoder}/{text_encoder}/{variant_name(use_local, use_global)}")
            continue
        seen.add(job)
        deduped.append(job)
    return deduped


def checkpoint_dir(vision_encoder: str, text_encoder: str, use_local: bool, use_global: bool) -> Path:
    return Path("checkpoints") / f"{vision_encoder}_{text_encoder}_{variant_name(use_local, use_global)}"


def final_checkpoint_path(vision_encoder: str, text_encoder: str, use_local: bool, use_global: bool) -> Path:
    return PROJECT_ROOT / checkpoint_dir(vision_encoder, text_encoder, use_local, use_global) / "final.pt"


def best_checkpoint_path(vision_encoder: str, text_encoder: str, use_local: bool, use_global: bool) -> Path:
    return PROJECT_ROOT / checkpoint_dir(vision_encoder, text_encoder, use_local, use_global) / "best.pt"


def filter_resume_completed_jobs(
    jobs: list[tuple[str, str, bool, bool]],
    results_path: Path,
    resume: bool,
) -> list[tuple[str, str, bool, bool]]:
    if not resume:
        return jobs
    pending: list[tuple[str, str, bool, bool]] = []
    for vision_encoder, text_encoder, use_local, use_global in jobs:
        variant = variant_name(use_local, use_global)
        final_checkpoint = final_checkpoint_path(vision_encoder, text_encoder, use_local, use_global)
        if final_checkpoint.exists() and result_row_exists(results_path, vision_encoder, text_encoder, use_local, use_global):
            print(f"Skipping completed job: {vision_encoder}/{text_encoder}/{variant}")
            continue
        pending.append((vision_encoder, text_encoder, use_local, use_global))
    return pending


def run_job(
    base_config: dict,
    results_path: Path,
    vision_encoder: str,
    text_encoder: str,
    use_local: bool,
    use_global: bool,
    fusion_type: str | None,
    resume: bool = False,
) -> None:
    config = deepcopy(base_config)
    model_cfg = config["model"]
    model_cfg["vision_encoder"] = vision_encoder
    model_cfg["text_encoder"] = text_encoder
    model_cfg["use_local_blf"] = use_local
    model_cfg["use_global_blf"] = use_global
    if fusion_type:
        model_cfg["fusion_type"] = fusion_type

    variant = variant_name(use_local, use_global)
    save_dir = checkpoint_dir(vision_encoder, text_encoder, use_local, use_global)
    config["training"]["save_dir"] = str(save_dir)

    final_checkpoint = final_checkpoint_path(vision_encoder, text_encoder, use_local, use_global)
    if resume and final_checkpoint.exists():
        if result_row_exists(results_path, vision_encoder, text_encoder, use_local, use_global):
            print(f"Skipping completed job: {vision_encoder}/{text_encoder}/{variant}")
            return
        best_checkpoint = best_checkpoint_path(vision_encoder, text_encoder, use_local, use_global)
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Completed run is missing its best checkpoint: {best_checkpoint}")
        metrics = load_checkpoint_metrics(best_checkpoint)
        row = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "vision_setup": setup_name(vision_encoder, use_local, use_global),
            "fusion_type": model_cfg.get("fusion_type", "concat_mlp"),
            "use_local_blf": use_local,
            "use_global_blf": use_global,
            **metrics,
        }
        append_result(results_path, row)
        print(f"Recovered best-checkpoint row from {best_checkpoint}")
        return

    metrics = train_from_config(config, resume=resume)
    row = {
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
        "vision_setup": setup_name(vision_encoder, use_local, use_global),
        "fusion_type": model_cfg.get("fusion_type", "concat_mlp"),
        "use_local_blf": use_local,
        "use_global_blf": use_global,
        **metrics,
    }
    append_result(results_path, row)
    print(f"Saved matrix row to {results_path}")


def merge_result_shards(final_path: Path, shard_paths: list[Path]) -> None:
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        with shard_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                append_result(final_path, row)


def run_parallel(args: argparse.Namespace, jobs: list[tuple[str, str, bool, bool]], results_path: Path) -> None:
    gpus = normalize_gpus(args.gpus)
    gpu_tokens = resolve_gpu_tokens(gpus)
    run_id = f"{int(time.time())}_{os.getpid()}"
    shard_dir = PROJECT_ROOT / "results" / ".alignment_matrix_shards" / run_id
    log_dir = PROJECT_ROOT / "logs" / "encoder_matrix" / run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(enumerate(jobs, start=1))
    active: list[dict[str, object]] = []
    available_gpus = list(gpus)
    completed_shards: list[Path] = []

    print(f"Running {len(jobs)} jobs across {len(gpus)} GPUs: {', '.join(gpus)}")
    print(f"Per-job logs: {log_dir}")

    try:
        while pending or active:
            while pending and available_gpus:
                job_index, (vision_encoder, text_encoder, use_local, use_global) = pending.pop(0)
                gpu = available_gpus.pop(0)
                variant = variant_name(use_local, use_global)
                shard_path = shard_dir / f"job_{job_index:03d}_{vision_encoder}_{text_encoder}_{variant}.csv"
                log_path = log_dir / f"job_{job_index:03d}_{vision_encoder}_{text_encoder}_{variant}.log"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    args.config,
                    "--mode",
                    args.mode,
                    "--matrix_variant",
                    variant,
                    "--results_path",
                    str(shard_path.relative_to(PROJECT_ROOT)),
                    "--vision_encoders",
                    vision_encoder,
                    "--text_encoders",
                    text_encoder,
                ]
                if args.fusion_type:
                    cmd.extend(["--fusion_type", args.fusion_type])
                if args.resume:
                    cmd.append("--resume")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_tokens[gpu]
                env["PYTHONUNBUFFERED"] = "1"
                log_handle = log_path.open("w")
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append(
                    {
                        "proc": proc,
                        "log_handle": log_handle,
                        "log_path": log_path,
                        "shard_path": shard_path,
                        "label": f"{vision_encoder}/{text_encoder}/{variant}",
                        "gpu": gpu,
                    }
                )
                print(f"[GPU {gpu}] started job {job_index}/{len(jobs)}: {vision_encoder}/{text_encoder}/{variant}")

            time.sleep(5)
            for item in active[:]:
                proc = item["proc"]
                assert isinstance(proc, subprocess.Popen)
                returncode = proc.poll()
                if returncode is None:
                    continue
                log_handle = item["log_handle"]
                assert hasattr(log_handle, "close")
                log_handle.close()
                active.remove(item)

                label = str(item["label"])
                gpu = str(item["gpu"])
                log_path = Path(item["log_path"])
                shard_path = Path(item["shard_path"])
                if returncode != 0:
                    raise RuntimeError(f"Job failed on GPU {gpu}: {label}. See {log_path}")

                completed_shards.append(shard_path)
                available_gpus.append(gpu)
                print(f"[GPU {gpu}] finished: {label}")
    except Exception:
        for item in active:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            proc.terminate()
            log_handle = item["log_handle"]
            assert hasattr(log_handle, "close")
            log_handle.close()
        raise

    merge_result_shards(results_path, completed_shards)
    print(f"Merged {len(completed_shards)} rows into {results_path}")


def main() -> None:
    args = parse_args()
    base_config = load_config(PROJECT_ROOT / args.config)
    results_path = PROJECT_ROOT / args.results_path
    jobs = filter_resume_completed_jobs(iter_jobs(args), results_path, resume=args.resume)
    gpus = normalize_gpus(args.gpus)

    if not jobs:
        print("No pending jobs.")
        return

    if len(gpus) > 1 and len(jobs) > 1:
        args.gpus = gpus
        run_parallel(args, jobs, results_path)
        return
    if len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_gpu_tokens(gpus)[gpus[0]]

    for vision_encoder, text_encoder, use_local, use_global in jobs:
        run_job(
            base_config,
            results_path,
            vision_encoder,
            text_encoder,
            use_local,
            use_global,
            args.fusion_type,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
