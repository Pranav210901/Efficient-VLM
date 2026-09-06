from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data import build_dataloaders
from src.models import PairedOpenCLIPModel
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text
from src.training.evaluate import evaluate_model
from src.training.train import build_model_from_config, train_from_config
from src.utils.checkpoint import load_checkpoint
from src.utils.config import deep_update, load_config
from src.utils.device import get_device
from src.utils.logging import count_parameters


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pipeline(path: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(path)
    if not value.is_absolute():
        value = project_root() / value
    if not value.exists():
        raise FileNotFoundError(value)
    return value, load_config(value)


def reference_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [dict(value) for value in pipeline.get("references", [])]
    for job in jobs:
        job["experiment_id"] = str(job["id"])
    return jobs


def unimodal_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for experiment in pipeline.get("unimodal_experiments", []):
        experiment_id = str(experiment["id"])
        for seed in experiment.get("seeds", [42]):
            jobs.append(
                {
                    **dict(experiment),
                    "seed": int(seed),
                    "experiment_id": experiment_id,
                    "run_id": f"{experiment_id}__seed_{int(seed)}",
                }
            )
    return jobs


def _job_index(value: int | None) -> int:
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if value is None else int(value)
    if selected < 0:
        raise ValueError("job index must be non-negative")
    return selected


def _select(jobs: list[dict[str, Any]], index: int | None) -> dict[str, Any]:
    selected = _job_index(index)
    if selected >= len(jobs):
        raise IndexError(f"job index {selected} is outside manifest size {len(jobs)}")
    return jobs[selected]


def build_unimodal_config(job: dict[str, Any]) -> dict[str, Any]:
    base_path = Path(str(job["base_config"]))
    if not base_path.is_absolute():
        base_path = project_root() / base_path
    config = deep_update(load_config(base_path), dict(job.get("overrides", {})))
    config["seed"] = int(job["seed"])
    config["experiment_id"] = str(job["experiment_id"])
    config["training"]["save_dir"] = f"checkpoints/alignment_v2/{job['run_id']}"
    return config


def validate_pipeline(pipeline_path: str | Path) -> dict[str, Any]:
    path, pipeline = load_pipeline(pipeline_path)
    references = reference_jobs(pipeline)
    unimodal = unimodal_jobs(pipeline)
    if not references:
        raise ValueError("pipeline must define at least one paired reference model")
    if not unimodal:
        raise ValueError("pipeline must define at least one frozen-unimodal run")
    ids = [job["experiment_id"] for job in references] + [job["run_id"] for job in unimodal]
    if len(ids) != len(set(ids)):
        raise ValueError("pipeline experiment/run IDs must be unique")
    if not all("openclip" in str(job.get("family", "openclip")).lower() for job in references):
        raise ValueError("the current reference runner supports paired OpenCLIP models")
    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip_torch is required before submitting alignment v2") from exc
    available_references = set(open_clip.list_pretrained())
    unavailable = [
        (str(job["model"]["name"]), str(job["model"]["pretrained"]))
        for job in references
        if (str(job["model"]["name"]), str(job["model"]["pretrained"])) not in available_references
    ]
    if unavailable:
        raise ValueError(f"OpenCLIP does not provide the configured reference checkpoints: {unavailable}")
    required_data: set[Path] = set()
    for job in references:
        config_path = Path(str(job["config"]))
        config_path = config_path if config_path.is_absolute() else project_root() / config_path
        config = load_config(config_path)
        required_data.update(project_root() / str(config["data"][key]) for key in ("train_csv", "val_csv"))
    for job in unimodal:
        config = build_unimodal_config(job)
        required_data.update(project_root() / str(config["data"][key]) for key in ("train_csv", "val_csv"))
        if not bool(config["model"].get("freeze_vision", True)) or not bool(config["model"].get("freeze_text", True)):
            raise ValueError(f"{job['run_id']} must keep both unimodal encoders frozen")
    missing = sorted(str(value) for value in required_data if not value.exists())
    if missing:
        raise FileNotFoundError(f"required datasets are missing: {missing}")
    manifest_root = project_root() / str(pipeline.get("output_root", "results/alignment_v2")) / "manifests"
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "array_index": index,
                    "experiment_id": job["experiment_id"],
                    "model_name": job["model"]["name"],
                    "pretrained": job["model"]["pretrained"],
                    "config": job["config"],
                }
                for index, job in enumerate(references)
            ]
        ),
        manifest_root / "reference_jobs.csv",
    )
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "array_index": index,
                    "experiment_id": job["experiment_id"],
                    "run_id": job["run_id"],
                    "seed": job["seed"],
                    "base_config": job["base_config"],
                    "checkpoint": f"checkpoints/alignment_v2/{job['run_id']}/best.pt",
                }
                for index, job in enumerate(unimodal)
            ]
        ),
        manifest_root / "unimodal_jobs.csv",
    )
    report = {
        "status": "READY",
        "pipeline": str(path),
        "reference_jobs": len(references),
        "unimodal_jobs": len(unimodal),
        "reference_array": f"0-{len(references) - 1}",
        "unimodal_array": f"0-{len(unimodal) - 1}",
        "frozen_unimodal_required": True,
    }
    atomic_json(report, manifest_root / "validation.json")
    return report


def prefetch_models(pipeline_path: str | Path) -> dict[str, Any]:
    """Populate the shared model caches once before concurrent GPU arrays."""
    _, pipeline = load_pipeline(pipeline_path)
    requested = [
        f"openclip:{job['model']['name']}:{job['model']['pretrained']}"
        for job in reference_jobs(pipeline)
    ]
    requested.extend(f"unimodal:{job['experiment_id']}" for job in unimodal_jobs(pipeline))
    requested = list(dict.fromkeys(requested))
    output = project_root() / str(pipeline.get("output_root", "results/alignment_v2"))
    marker = output / "prefetch.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("status") == "COMPLETED" and previous.get("models") == requested:
            return {**previous, "status": "SKIPPED_ALREADY_COMPLETE"}
    loaded: list[str] = []
    for job in reference_jobs(pipeline):
        model = PairedOpenCLIPModel(str(job["model"]["name"]), str(job["model"]["pretrained"]))
        loaded.append(f"openclip:{job['model']['name']}:{job['model']['pretrained']}")
        del model
        gc.collect()
    seen: set[str] = set()
    for job in unimodal_jobs(pipeline):
        experiment_id = str(job["experiment_id"])
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        config = build_unimodal_config(job)
        model = build_model_from_config(config)
        loaded.append(f"unimodal:{experiment_id}")
        del model
        gc.collect()
    result = {"status": "COMPLETED", "models": loaded, "unique_models": len(loaded)}
    atomic_json(result, output / "prefetch.json")
    return result


@torch.no_grad()
def profile_model(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    benchmark = config.get("benchmark", {})
    warmup = int(benchmark.get("warmup", 5))
    iterations = int(benchmark.get("iterations", 20))
    image_size = int(config["data"].get("image_size", 224))
    images = torch.randn(int(benchmark.get("image_batch_size", 1)), 3, image_size, image_size, device=device)
    captions = ["a photograph of an everyday object"] * int(benchmark.get("text_batch_size", 1))

    def synchronise() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model.encode_image(images)
        model.encode_text(captions)
    synchronise()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iterations):
        model.encode_image(images)
    synchronise()
    image_ms = (time.perf_counter() - started) * 1000.0 / max(1, iterations * len(images))
    started = time.perf_counter()
    for _ in range(iterations):
        model.encode_text(captions)
    synchronise()
    text_ms = (time.perf_counter() - started) * 1000.0 / max(1, iterations * len(captions))
    parameters = count_parameters(model)
    return {
        **parameters,
        "image_query_latency_ms": image_ms,
        "text_query_latency_ms": text_ms,
        "bidirectional_pair_latency_ms": image_ms + text_ms,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
    }


def _result_row(kind: str, experiment_id: str, seed: int | None, metrics: dict[str, float], efficiency: dict[str, float]) -> dict[str, Any]:
    return {
        "kind": kind,
        "experiment_id": experiment_id,
        "seed": seed,
        **{key: float(value) for key, value in metrics.items()},
        **{key: float(value) for key, value in efficiency.items()},
    }


def evaluate_reference(pipeline_path: str | Path, index: int | None = None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(reference_jobs(pipeline), index)
    destination = project_root() / str(pipeline.get("output_root", "results/alignment_v2")) / "reference" / str(job["experiment_id"])
    existing = destination / "metrics.json"
    if existing.exists():
        return {**json.loads(existing.read_text()), "status": "SKIPPED_ALREADY_COMPLETE"}
    config_path = Path(str(job["config"]))
    config_path = config_path if config_path.is_absolute() else project_root() / config_path
    config = load_config(config_path)
    device = get_device(str(config.get("device", "auto")))
    model = PairedOpenCLIPModel(str(job["model"]["name"]), str(job["model"]["pretrained"])).to(device)
    # The paired model is an inference-only checkpoint-native reference. Mark
    # it frozen so params_trainable reflects the actual experiment policy
    # rather than the library's default requires_grad flags.
    model.requires_grad_(False)
    model.eval()
    _, loader = build_dataloaders(config)
    metrics = evaluate_model(model, loader, device, list(config.get("evaluation", {}).get("k_values", [1, 5, 10])))
    efficiency = profile_model(model, config, device)
    row = _result_row("paired_reference", str(job["experiment_id"]), None, metrics, efficiency)
    atomic_json(row, destination / "metrics.json")
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    return row


def train_unimodal(pipeline_path: str | Path, index: int | None = None, resume: bool = True) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(unimodal_jobs(pipeline), index)
    config = build_unimodal_config(job)
    checkpoint_root = project_root() / str(config["training"]["save_dir"])
    if resume and (checkpoint_root / "best.pt").exists() and (checkpoint_root / "final.pt").exists():
        return {"status": "SKIPPED_ALREADY_COMPLETE", "run_id": job["run_id"]}
    metrics = train_from_config(config, resume=resume)
    return {"status": "COMPLETED", "run_id": job["run_id"], **metrics}


def evaluate_unimodal(pipeline_path: str | Path, index: int | None = None) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    job = _select(unimodal_jobs(pipeline), index)
    config = build_unimodal_config(job)
    destination = project_root() / str(pipeline.get("output_root", "results/alignment_v2")) / "unimodal" / str(job["run_id"])
    existing = destination / "metrics.json"
    if existing.exists():
        return {**json.loads(existing.read_text()), "status": "SKIPPED_ALREADY_COMPLETE"}
    checkpoint = project_root() / str(config["training"]["save_dir"]) / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"training dependency is incomplete: {checkpoint}")
    device = get_device(str(config.get("device", "auto")))
    model = build_model_from_config(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    _, loader = build_dataloaders(config)
    metrics = evaluate_model(model, loader, device, list(config.get("evaluation", {}).get("k_values", [1, 5, 10])))
    efficiency = profile_model(model, config, device)
    row = _result_row("frozen_unimodal", str(job["experiment_id"]), int(job["seed"]), metrics, efficiency)
    row["run_id"] = str(job["run_id"])
    row["checkpoint"] = str(checkpoint.relative_to(project_root()))
    atomic_json(row, destination / "metrics.json")
    atomic_csv(pd.DataFrame([row]), destination / "metrics.csv")
    return row


def build_report(pipeline_path: str | Path) -> dict[str, Any]:
    _, pipeline = load_pipeline(pipeline_path)
    output = project_root() / str(pipeline.get("output_root", "results/alignment_v2"))
    expected_reference = reference_jobs(pipeline)
    expected_unimodal = unimodal_jobs(pipeline)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in expected_reference:
        path = output / "reference" / str(job["experiment_id"]) / "metrics.json"
        if not path.exists():
            missing.append(str(path.relative_to(project_root())))
        else:
            rows.append(json.loads(path.read_text()))
    for job in expected_unimodal:
        path = output / "unimodal" / str(job["run_id"]) / "metrics.json"
        if not path.exists():
            missing.append(str(path.relative_to(project_root())))
        else:
            rows.append(json.loads(path.read_text()))
    if not rows:
        raise RuntimeError("no v2 evaluation results are available")
    frame = pd.DataFrame(rows)
    atomic_csv(frame, output / "results_long.csv")
    numeric = [
        column
        for column in ("i2t_R@1", "t2i_R@1", "mean_R@1", "bidirectional_pair_latency_ms", "params_total", "params_trainable")
        if column in frame
    ]
    summary = (
        frame.groupby(["kind", "experiment_id"], dropna=False)[numeric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["__".join(value).strip("_") if isinstance(value, tuple) else str(value) for value in summary.columns]
    atomic_csv(summary, output / "results_summary.csv")
    display_columns = [
        column
        for column in (
            "kind",
            "experiment_id",
            "seed",
            "i2t_R@1",
            "t2i_R@1",
            "mean_R@1",
            "bidirectional_pair_latency_ms",
            "params_total",
            "params_trainable",
        )
        if column in frame
    ]
    markdown = (
        "# Alignment v2 results\n\n"
        f"Status: **{'INCOMPLETE' if missing else 'COMPLETE'}**\n\n"
        "All retrieval metrics use the same grouped five-caption COCO validation protocol. "
        "Latency is measured independently for batch-one image and text queries on the executing GPU.\n\n"
        + frame[display_columns].sort_values(["kind", "experiment_id", "seed"], na_position="first").to_markdown(index=False)
        + "\n"
    )
    if missing:
        markdown += "\n## Missing evaluations\n\n" + "\n".join(f"- `{value}`" for value in missing) + "\n"
    atomic_text(markdown, output / "report.md")
    report = {"status": "INCOMPLETE" if missing else "COMPLETE", "result_rows": len(frame), "missing": missing}
    atomic_json(report, output / "report.json")
    return report


def status(pipeline_path: str | Path) -> pd.DataFrame:
    _, pipeline = load_pipeline(pipeline_path)
    output = project_root() / str(pipeline.get("output_root", "results/alignment_v2"))
    rows = []
    for job in reference_jobs(pipeline):
        target = output / "reference" / str(job["experiment_id"]) / "metrics.json"
        rows.append({"stage": "reference", "run_id": job["experiment_id"], "complete": target.exists(), "target": str(target)})
    for job in unimodal_jobs(pipeline):
        checkpoint = project_root() / f"checkpoints/alignment_v2/{job['run_id']}/best.pt"
        result = output / "unimodal" / str(job["run_id"]) / "metrics.json"
        rows.append({"stage": "train", "run_id": job["run_id"], "complete": checkpoint.exists(), "target": str(checkpoint)})
        rows.append({"stage": "evaluate", "run_id": job["run_id"], "complete": result.exists(), "target": str(result)})
    return pd.DataFrame(rows)


def main() -> None:
    os.chdir(project_root())
    parser = argparse.ArgumentParser(description="Run alignment-v2 reference and frozen-unimodal experiments")
    parser.add_argument("command", choices=["validate", "prefetch", "reference", "train", "evaluate", "report", "status"])
    parser.add_argument("--pipeline", default="configs/alignment_v2/pipeline.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        result: Any = validate_pipeline(args.pipeline)
    elif args.command == "prefetch":
        result = prefetch_models(args.pipeline)
    elif args.command == "reference":
        result = evaluate_reference(args.pipeline, args.index)
    elif args.command == "train":
        result = train_unimodal(args.pipeline, args.index, resume=not args.no_resume)
    elif args.command == "evaluate":
        result = evaluate_unimodal(args.pipeline, args.index)
    elif args.command == "report":
        result = build_report(args.pipeline)
    else:
        result = status(args.pipeline)
    print(result.to_string(index=False) if isinstance(result, pd.DataFrame) else result)


if __name__ == "__main__":
    main()
