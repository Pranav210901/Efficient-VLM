from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.multitask.checkpoint_selection import VARIANTS, resolve_best_checkpoints
from src.multitask.config import DEFAULT_DATASETS, STEP_11_5_BLF_CONFIG_IDS, VISION_ENCODERS
from src.multitask.datasets import validate_datasets
from src.multitask.runner import _write_tables, run_multitask_evaluation
from src.utils.gpu_jobs import normalize_gpu_ids, resolve_gpu_tokens


MODE_RANK = {"smoke": 1, "full": 2}
MERGED_MARKER = ".merged_into_canonical"


def normalize_gpus(values: list[str] | None) -> list[str]:
    return normalize_gpu_ids(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate independent Phase 1 configurations with one worker per GPU.")
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--config_scope", choices=("baseline_all_pairs", "baseline_plus_reliable_blf"), default="baseline_all_pairs")
    parser.add_argument(
        "--config_ids",
        nargs="+",
        default=None,
        help="Evaluate only these exact canonical configuration IDs.",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--split_role", choices=("historical", "development", "final_test"), default="historical")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _mode_from_cache(value: Any) -> str | None:
    if pd.isna(value):
        return None
    name = Path(str(value)).name
    for mode in MODE_RANK:
        if name.endswith(f"__{mode}.pt"):
            return mode
    return None


def _with_manifest_modes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "mode" not in result:
        result["mode"] = None
    invalid = ~result["mode"].isin(MODE_RANK)
    if "cache" in result:
        result.loc[invalid, "mode"] = result.loc[invalid, "cache"].map(_mode_from_cache)
    return result


def _with_result_modes(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "mode" not in result:
        result["mode"] = None
    if manifest.empty or not {"config_id", "task", "mode"}.issubset(manifest.columns):
        return result
    mode_by_task = {
        (str(row.config_id), str(row.task)): row.mode
        for row in manifest.itertuples()
        if row.mode in MODE_RANK
    }
    invalid = ~result["mode"].isin(MODE_RANK)
    result.loc[invalid, "mode"] = [
        mode_by_task.get((str(row.config_id), str(row.task)))
        for row in result.loc[invalid].itertuples()
    ]
    return result


def _preferred_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["_mode_rank"] = ranked.get("mode", pd.Series(index=ranked.index, dtype=object)).map(MODE_RANK).fillna(0)
    ranked["_source_order"] = range(len(ranked))
    ranked = ranked.sort_values(["_mode_rank", "_source_order"], kind="stable")
    return ranked.drop_duplicates(keys, keep="last").drop(columns=["_mode_rank", "_source_order"])


def completed_configs(output: Path, datasets: list[str], mode: str, split_role: str = "historical") -> set[str]:
    manifest_path = output / "evaluation_manifest.csv"
    if not manifest_path.exists(): return set()
    frame = _with_manifest_modes(read_table(manifest_path))
    ready = {row["task"] for row in validate_datasets(PROJECT_ROOT, datasets, split_role=split_role) if row["status"] == "ready"}
    if frame.empty or not ready: return set()
    required_rank = MODE_RANK[mode]
    ranks = frame["mode"].map(MODE_RANK).fillna(0)
    complete = frame[frame.status.eq("complete") & frame.task.isin(ready) & ranks.ge(required_rank)]
    counts = complete.groupby("config_id").task.nunique()
    return set(counts[counts >= len(ready)].index)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def worker(args: argparse.Namespace) -> None:
    if not args.worker_output: raise ValueError("--worker_output is required")
    print(f"Worker starting configuration {args.worker_config} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    run_multitask_evaluation(
        PROJECT_ROOT, args.datasets, config_scope=args.config_scope, mode=args.mode, device="cuda",
        config_ids=[args.worker_config], output_root_override=PROJECT_ROOT / args.worker_output, split_role=args.split_role,
    )


def select_checkpoint_specs(
    checkpoint_root: str | Path,
    config_scope: str,
    config_ids: list[str] | tuple[str, ...] | None = None,
):
    """Resolve the exact coordinator allow-list without expanding BLF runs."""
    requested = set(config_ids) if config_ids else None
    if requested:
        requested_variants = {value.rsplit("__", 1)[-1] for value in requested}
        unknown_variants = requested_variants - set(VARIANTS)
        if unknown_variants:
            raise ValueError(f"Unknown variants in configuration IDs: {sorted(unknown_variants)}")
        variants = tuple(value for value in VARIANTS if value in requested_variants)
    elif config_scope == "baseline_plus_reliable_blf":
        variants = VARIANTS
        baseline_specs = resolve_best_checkpoints(
            checkpoint_root, list(VISION_ENCODERS), ("baseline",), load_epochs=False
        )
        requested = {spec.config_id for spec in baseline_specs} | set(STEP_11_5_BLF_CONFIG_IDS[:2])
    else:
        variants = ("baseline",)

    specs = resolve_best_checkpoints(
        checkpoint_root,
        list(VISION_ENCODERS),
        variants,
        config_ids=requested,
        load_epochs=False,
    )
    if requested is not None:
        missing = requested - {spec.config_id for spec in specs}
        if missing:
            raise KeyError(f"Unknown or unavailable configuration IDs: {sorted(missing)}")
    return specs


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_workers(worker_outputs: list[Path], canonical: Path) -> None:
    existing_manifest = _with_manifest_modes(read_table(canonical / "evaluation_manifest.csv"))
    existing_long = _with_result_modes(read_table(canonical / "task_results_long.csv"), existing_manifest)
    long_frames = [existing_long]
    manifest_frames = [existing_manifest]
    status_frames = [read_table(canonical / "task_status.csv")]
    for worker_output in worker_outputs:
        worker_manifest = _with_manifest_modes(read_table(worker_output / "evaluation_manifest.csv"))
        worker_long = _with_result_modes(read_table(worker_output / "task_results_long.csv"), worker_manifest)
        long_frames.append(worker_long)
        manifest_frames.append(worker_manifest)
        status_frames.append(read_table(worker_output / "task_status.csv"))
        for directory in ("cache", "predictions"):
            target = canonical / directory; target.mkdir(parents=True, exist_ok=True)
            for source in (worker_output / directory).glob("*"):
                if not source.is_file():
                    continue
                destination = target / source.name
                _atomic_copy(source, destination)
    nonempty_long = [value for value in long_frames if not value.empty]
    nonempty_manifest = [value for value in manifest_frames if not value.empty]
    long = _preferred_rows(pd.concat(nonempty_long, ignore_index=True), ["config_id", "task", "metric"]) if nonempty_long else pd.DataFrame()
    manifest = _preferred_rows(pd.concat(nonempty_manifest, ignore_index=True), ["config_id", "task"]) if nonempty_manifest else pd.DataFrame()
    for column, directory in (("cache", "cache"), ("predictions", "predictions"), ("predictions_t2i", "predictions"), ("per_class", "predictions")):
        if column in manifest:
            manifest[column] = manifest[column].map(
                lambda value: str(canonical / directory / Path(str(value)).name) if pd.notna(value) and str(value) else value
            )
    nonempty_status = [value for value in status_frames if not value.empty]
    status = pd.concat(nonempty_status, ignore_index=True).drop_duplicates("task", keep="last") if nonempty_status else pd.DataFrame(validate_datasets(PROJECT_ROOT, list(DEFAULT_DATASETS)))
    _write_tables(canonical, long.to_dict("records"), status.to_dict("records"), manifest.to_dict("records"))
    configurations = manifest.config_id.nunique() if "config_id" in manifest else 0
    print(f"Merged {configurations} configurations into {canonical}")


def recover_worker_outputs(canonical: Path, worker_outputs: list[Path] | None = None) -> int:
    candidates = worker_outputs
    if candidates is None:
        candidates = sorted((canonical / ".workers").glob("*/job_*"))
    pending = [
        value for value in candidates
        if value.is_dir() and not (value / MERGED_MARKER).exists()
    ]
    if not pending:
        return 0
    print(f"Recovering {len(pending)} unmerged worker outputs before scheduling jobs.", flush=True)
    merge_workers(pending, canonical)
    for output in pending:
        (output / MERGED_MARKER).write_text(f"merged_at={time.time()}\n")
    return len(pending)


@contextmanager
def coordinator_lock(canonical: Path):
    lock_path = canonical / ".coordinator.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner details unavailable"
            raise RuntimeError(f"Another multi-task coordinator is already running ({owner})") from error
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
    watched = [signal.SIGTERM, signal.SIGHUP]
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


def coordinator(args: argparse.Namespace) -> None:
    gpus = normalize_gpus(args.gpus)
    if not gpus: raise ValueError("At least one GPU must be supplied with --gpus")
    if args.mode == "smoke" and len(gpus) > 1:
        print("Smoke mode intentionally uses one GPU; extra visible GPUs will remain idle.")
        gpus = gpus[:1]
    gpu_tokens = resolve_gpu_tokens(gpus)
    canonical = PROJECT_ROOT / ("results/phase1_multitask" if args.split_role == "historical" else f"results/phase1_multitask_{args.split_role}")
    canonical.mkdir(parents=True, exist_ok=True)
    with coordinator_lock(canonical), recoverable_termination_signals():
        if args.resume:
            recover_worker_outputs(canonical)

        specs = select_checkpoint_specs(
            PROJECT_ROOT / "checkpoints", args.config_scope, args.config_ids
        )
        done = completed_configs(canonical, args.datasets, args.mode, args.split_role) if args.resume else set()
        selected_specs = specs[:1] if args.mode == "smoke" else specs
        jobs = [spec.config_id for spec in selected_specs if spec.config_id not in done]
        selected_ids = {spec.config_id for spec in selected_specs}
        for config_id in sorted(done & selected_ids):
            print("Skipping completed evaluation:", config_id)
        if not jobs:
            print("No pending multi-task evaluation jobs.")
            return

        run_id = f"{int(time.time())}_{os.getpid()}"
        worker_root = canonical / ".workers" / run_id
        log_root = PROJECT_ROOT / "logs/multitask" / run_id
        worker_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        pending = list(enumerate(jobs, 1))
        available = list(gpus)
        active: list[dict[str, Any]] = []
        recoverable: list[Path] = []
        print(f"Running {len(jobs)} configuration jobs across {len(gpus)} GPUs: {', '.join(gpus)}", flush=True)

        def stream_log(item: dict[str, Any]) -> None:
            path = Path(item["log"])
            if not path.exists(): return
            with path.open(errors="replace") as handle:
                handle.seek(int(item.get("log_offset", 0)))
                chunk = handle.read()
                item["log_offset"] = handle.tell()
            if chunk:
                prefix = f"[GPU {item['gpu']} | {item['config']}] "
                for line in chunk.splitlines(): print(prefix + line, flush=True)

        def stop_active_workers() -> None:
            for item in active:
                process = item["process"]
                if process.poll() is None:
                    process.terminate()
            for item in active:
                process = item["process"]
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                stream_log(item)
                if not item["handle"].closed:
                    item["handle"].close()
                recoverable.append(item["output"])

        try:
            while pending or active:
                while pending and available:
                    index, config_id = pending.pop(0)
                    gpu = available.pop(0)
                    output = worker_root / f"job_{index:03d}"
                    log = log_root / f"job_{index:03d}_{config_id}.log"
                    if args.resume:
                        worker_cache = output / "cache"
                        worker_cache.mkdir(parents=True, exist_ok=True)
                        for source in (canonical / "cache").glob(f"{config_id}__*"):
                            _atomic_copy(source, worker_cache / source.name)
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--mode", args.mode,
                        "--split_role", args.split_role,
                        "--config_scope", args.config_scope,
                        "--datasets", *args.datasets,
                        "--worker_config", config_id,
                        "--worker_output", str(output.relative_to(PROJECT_ROOT)),
                    ]
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
                            "config": config_id,
                            "output": output,
                            "log": log,
                            "log_offset": 0,
                            "last_heartbeat": time.monotonic(),
                        }
                    )
                    print(f"[GPU {gpu}] started {index}/{len(jobs)}: {config_id}; log={log}", flush=True)

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
                    active.remove(item)
                    available.append(item["gpu"])
                    if code != 0:
                        recoverable.append(item["output"])
                        raise RuntimeError(f"Evaluation failed for {item['config']}. See {item['log']}")

                    recoverable.append(item["output"])
                    merge_workers([item["output"]], canonical)
                    (item["output"] / MERGED_MARKER).write_text(f"merged_at={time.time()}\n")
                    recoverable.remove(item["output"])
                    print(f"[GPU {item['gpu']}] finished and committed {item['config']}", flush=True)
        except BaseException:
            stop_active_workers()
            unmerged = [value for value in recoverable if not (value / MERGED_MARKER).exists()]
            if unmerged:
                recover_worker_outputs(canonical, unmerged)
            print("Interrupted run recovered all completed tasks and valid caches before exiting.", flush=True)
            raise


def main() -> None:
    args = parse_args()
    if args.worker_config: worker(args)
    else: coordinator(args)


if __name__ == "__main__": main()
