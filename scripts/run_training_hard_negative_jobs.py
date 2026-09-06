#!/usr/bin/env python3
"""Run Phase 1.5 training-split hard-negative mining with one pair per GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.phase15.io_utils import atomic_json
from src.phase15.training_hard_negative_mining import (
    TRAINING_HARD_NEGATIVES_PER_POSITIVE,
    finalize_training_hard_negative_mining,
    mine_training_hard_negatives_for_config,
    prepare_training_hard_negative_sources,
    selected_training_pair_ids,
    training_configuration_status,
)
from src.utils.gpu_jobs import normalize_gpu_ids, resolve_gpu_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine training-split hard negatives with one selected configuration per GPU."
    )
    parser.add_argument("--gpus", nargs="+", required=False, help="Logical allocated GPU IDs, e.g. --gpus 0 1 2 3")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--count", type=int, default=TRAINING_HARD_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


@contextmanager
def coordinator_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".coordinator.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner details unavailable"
            raise RuntimeError(f"Another Step 16 coordinator is already running ({owner})") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} host={socket.gethostname()} started={time.time()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def recoverable_termination_signals():
    watched = (signal.SIGTERM, signal.SIGHUP)
    previous = {value: signal.getsignal(value) for value in watched}

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        for value in watched:
            signal.signal(value, interrupt)
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def worker(args: argparse.Namespace) -> None:
    print(
        f"Worker starting {args.worker_config} on CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    result = mine_training_hard_negatives_for_config(
        PROJECT_ROOT,
        args.worker_config,
        mode=args.mode,
        count=args.count,
        chunk_size=args.chunk_size,
        device="cuda:0",
        resume=args.resume,
    )
    print(json.dumps(result, indent=2), flush=True)


def coordinator(args: argparse.Namespace) -> None:
    if args.count < 1 or args.chunk_size < 1:
        raise ValueError("--count and --chunk-size must be positive")
    gpus = normalize_gpu_ids(args.gpus)
    if not gpus:
        raise ValueError("At least one allocated GPU must be supplied with --gpus")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU IDs must be unique: {gpus}")
    if args.mode == "smoke" and len(gpus) > 1:
        print("Smoke mode uses only the first GPU.", flush=True)
        gpus = gpus[:1]

    output = PROJECT_ROOT / "results/phase15/training_hard_negatives"
    state_path = output / "coordinator_state.json"
    with coordinator_lock(output), recoverable_termination_signals():
        sources = prepare_training_hard_negative_sources(PROJECT_ROOT)
        config_ids = selected_training_pair_ids(PROJECT_ROOT, args.mode)
        statuses = {
            configuration_id: training_configuration_status(
                PROJECT_ROOT,
                configuration_id,
                mode=args.mode,
                count=args.count,
                sources=sources,
            )
            for configuration_id in config_ids
        } if args.resume else {}
        pending = [
            configuration_id
            for configuration_id in config_ids
            if not statuses.get(configuration_id, {}).get("complete", False)
        ]
        for configuration_id in config_ids:
            status = statuses.get(configuration_id)
            if status is None:
                continue
            if status["complete"]:
                print(f"Skipping complete configuration: {configuration_id}", flush=True)
            elif status["retrieval_complete"] or status["classification_complete"]:
                print(
                    f"Partial resume {configuration_id}: "
                    f"retrieval={'ready' if status['retrieval_complete'] else 'pending'}, "
                    f"classification={'ready' if status['classification_complete'] else 'pending'}",
                    flush=True,
                )

        gpu_tokens = resolve_gpu_tokens(gpus)
        run_id = f"{int(time.time())}_{os.getpid()}"
        log_root = PROJECT_ROOT / "logs/phase15/hard_negative_mining" / run_id
        log_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            {
                "status": "running",
                "run_id": run_id,
                "mode": args.mode,
                "count": args.count,
                "configurations": config_ids,
                "pending": pending,
                "gpus": gpus,
                "logs": str(log_root),
            },
            state_path,
        )
        print(
            f"Running {len(pending)} pending configuration(s) across {min(len(gpus), len(pending))} GPU(s): "
            f"{', '.join(gpus)}",
            flush=True,
        )
        print(f"Per-worker logs: {log_root}", flush=True)

        queue = list(enumerate(pending, start=1))
        available = list(gpus)
        active: list[dict[str, Any]] = []

        def stream_log(item: dict[str, Any]) -> None:
            path = Path(item["log"])
            if not path.exists():
                return
            with path.open(errors="replace") as handle:
                handle.seek(int(item.get("log_offset", 0)))
                content = handle.read()
                item["log_offset"] = handle.tell()
            if content:
                prefix = f"[GPU {item['gpu']} | {item['config']}] "
                for line in content.splitlines():
                    print(prefix + line, flush=True)

        def stop_active_workers() -> None:
            for item in active:
                process = item["process"]
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 45
            while any(item["process"].poll() is None for item in active) and time.monotonic() < deadline:
                time.sleep(0.25)
            for item in active:
                process = item["process"]
                if process.poll() is None:
                    process.kill()
                process.wait()
                stream_log(item)
                if not item["handle"].closed:
                    item["handle"].close()

        try:
            while queue or active:
                while queue and available:
                    index, configuration_id = queue.pop(0)
                    gpu = available.pop(0)
                    log = log_root / f"job_{index:03d}_{configuration_id}.log"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--mode",
                        args.mode,
                        "--count",
                        str(args.count),
                        "--chunk-size",
                        str(args.chunk_size),
                        "--worker-config",
                        configuration_id,
                    ]
                    if args.resume:
                        command.append("--resume")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_tokens[gpu]
                    env["PYTHONUNBUFFERED"] = "1"
                    handle = log.open("w")
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    active.append(
                        {
                            "process": process,
                            "handle": handle,
                            "gpu": gpu,
                            "config": configuration_id,
                            "log": log,
                            "log_offset": 0,
                            "last_heartbeat": time.monotonic(),
                        }
                    )
                    print(f"[GPU {gpu}] started {index}/{len(pending)}: {configuration_id}", flush=True)

                time.sleep(2)
                for item in active[:]:
                    stream_log(item)
                    code = item["process"].poll()
                    if code is None:
                        if time.monotonic() - float(item["last_heartbeat"]) >= 30:
                            print(f"[GPU {item['gpu']}] still running: {item['config']}", flush=True)
                            item["last_heartbeat"] = time.monotonic()
                        continue
                    if not item["handle"].closed:
                        item["handle"].close()
                    stream_log(item)
                    active.remove(item)
                    available.append(item["gpu"])
                    if code != 0:
                        raise RuntimeError(
                            f"Mining failed for {item['config']} on GPU {item['gpu']}. See {item['log']}"
                        )
                    print(f"[GPU {item['gpu']}] finished: {item['config']}", flush=True)
        except BaseException as error:
            stop_active_workers()
            atomic_json(
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "run_id": run_id,
                    "error": f"{type(error).__name__}: {error}",
                    "logs": str(log_root),
                    "resume_safe": True,
                },
                state_path,
            )
            print("Step 16 stopped; completed caches and per-configuration shards remain resumable.", flush=True)
            raise

        try:
            result = finalize_training_hard_negative_mining(
                PROJECT_ROOT,
                mode=args.mode,
                count=args.count,
                config_ids=config_ids,
            )
        except BaseException as error:
            atomic_json(
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "run_id": run_id,
                    "stage": "finalization",
                    "error": f"{type(error).__name__}: {error}",
                    "logs": str(log_root),
                    "resume_safe": True,
                },
                state_path,
            )
            raise
        atomic_json({"status": "complete", "run_id": run_id, "logs": str(log_root), **result}, state_path)
        print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    with recoverable_termination_signals():
        if args.worker_config:
            worker(args)
        else:
            coordinator(args)


if __name__ == "__main__":
    main()
