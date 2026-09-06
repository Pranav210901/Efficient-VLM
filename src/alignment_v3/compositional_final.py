"""One-shot SugarCrepe and Winoground evaluation for the frozen final roster."""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.alignment_v3.efficiency_frontier import (
    _reference_weight_manifest,
    parameter_counts_reference,
)
from src.alignment_v3.final_zero_shot import (
    ROOT,
    _hardware_guard,
    _jobs,
    _load_student,
    _pipeline,
    _student_transform,
)
from src.alignment_v3.fingerprint import hash_config, hash_payload, sha256_file
from src.alignment_v3.references import build_reference
from src.multitask.compositional_evaluator import sugarcrepe_scores, winoground_scores
from src.phase15.io_utils import atomic_csv, atomic_json, atomic_text

DEFAULT_PIPELINE = "configs/compositional_final/pipeline.yaml"
PREREGISTRATION = ROOT / "configs/compositional_final/preregistration.json"


def _output(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _results_exist(pipeline: dict[str, Any]) -> bool:
    root = _output(pipeline) / "per_run"
    return root.exists() and any(root.rglob("evaluation.json"))


def acquire_sugarcrepe(pipeline: dict[str, Any]) -> dict[str, Any]:
    if (_output(pipeline) / "manifests/data_manifest.json").exists() or _results_exist(pipeline):
        raise RuntimeError("dataset roster is already frozen; refusing post-freeze acquisition")
    spec = pipeline["datasets"]["sugarcrepe"]
    root = ROOT / str(spec["annotation_root"])
    rows = []
    for category in spec["categories"]:
        destination = root / f"{category}.json"
        if not destination.is_file():
            url = f"{str(spec['official_raw_base']).rstrip('/')}/{category}.json"
            with urllib.request.urlopen(url, timeout=60) as response:
                _atomic_bytes(destination, response.read())
        payload = json.loads(destination.read_text())
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"invalid SugarCrepe annotation file: {destination}")
        required = {"filename", "caption", "negative_caption"}
        for key, value in payload.items():
            if not required.issubset(value):
                raise ValueError(f"SugarCrepe {category}/{key} lacks {sorted(required)}")
        rows.append({"category": category, "examples": len(payload), "sha256": sha256_file(destination)})
    return {"status": "READY", "source": "official_RAIVNLab_repository", "categories": rows}


def acquire_winoground(pipeline: dict[str, Any], *, acknowledge_license: bool) -> dict[str, Any]:
    if not acknowledge_license:
        raise RuntimeError(
            "Winoground is gated. Read and accept its terms at the configured license URL, "
            "authenticate with Hugging Face, then pass --acknowledge-winoground-license."
        )
    if (_output(pipeline) / "manifests/data_manifest.json").exists() or _results_exist(pipeline):
        raise RuntimeError("dataset roster is already frozen; refusing post-freeze acquisition")
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for Winoground acquisition") from exc
    spec = pipeline["datasets"]["winoground"]
    target = ROOT / str(spec["parquet"])
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=str(spec["repository"]),
                repo_type="dataset",
                filename=str(spec["repository_file"]),
                local_dir=target.parents[1],
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Winoground download failed. Accept access at {spec['license_url']} and run `hf auth login`."
        ) from exc
    if downloaded.resolve() != target.resolve():
        _atomic_bytes(target, downloaded.read_bytes())
    return {"status": "READY", "path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def _sugarcrepe_rows(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    spec = pipeline["datasets"]["sugarcrepe"]
    annotation_root = ROOT / str(spec["annotation_root"])
    image_root = ROOT / str(spec["image_root"])
    rows: list[dict[str, str]] = []
    for category in spec["categories"]:
        path = annotation_root / f"{category}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing SugarCrepe annotation: {path}")
        payload = json.loads(path.read_text())
        for key in sorted(payload, key=lambda value: int(value) if str(value).isdigit() else str(value)):
            value = payload[key]
            image = image_root / str(value["filename"])
            if not image.is_file():
                raise FileNotFoundError(f"missing SugarCrepe COCO image: {image}")
            rows.append(
                {
                    "sample_id": f"{category}:{key}",
                    "category": str(category),
                    "image_path": str(image.resolve()),
                    "positive_caption": str(value["caption"]),
                    "negative_caption": str(value["negative_caption"]),
                }
            )
    if not rows:
        raise ValueError("SugarCrepe contains no examples")
    return rows


def _winoground_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for Winoground parquet evaluation") from exc
    rows = pq.read_table(path).to_pylist()
    required = {"caption_0", "caption_1", "image_0", "image_1"}
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"Winoground row {index} lacks {sorted(required)}")
    if not rows:
        raise ValueError("Winoground contains no examples")
    return rows


def _dataset_identities(pipeline: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    identities: dict[str, Any] = {}
    tasks: list[str] = []
    sugar_rows = _sugarcrepe_rows(pipeline)
    sugar_spec = pipeline["datasets"]["sugarcrepe"]
    annotation_paths = [
        ROOT / str(sugar_spec["annotation_root"]) / f"{category}.json"
        for category in sugar_spec["categories"]
    ]
    identities["sugarcrepe"] = {
        "samples": len(sugar_rows),
        "categories": list(sugar_spec["categories"]),
        "annotation_sha256": {path.name: sha256_file(path) for path in annotation_paths},
        "semantic_sha256": hash_payload(
            [(row["sample_id"], row["image_path"], row["positive_caption"], row["negative_caption"]) for row in sugar_rows]
        ),
    }
    tasks.append("sugarcrepe")
    win_path = ROOT / str(pipeline["datasets"]["winoground"]["parquet"])
    if win_path.is_file():
        win_rows = _winoground_rows(win_path)
        identities["winoground"] = {
            "samples": len(win_rows),
            "parquet_sha256": sha256_file(win_path),
            "schema": sorted(win_rows[0]),
        }
        tasks.append("winoground")
    return identities, tasks


def freeze(pipeline: dict[str, Any]) -> dict[str, Any]:
    if _results_exist(pipeline):
        raise RuntimeError("evaluation results already exist; the frozen data roster cannot be changed")
    identities, tasks = _dataset_identities(pipeline)
    payload = {
        "status": "FROZEN",
        "tasks": tasks,
        "primary_task": "sugarcrepe",
        "datasets": identities,
        "roster_sha256": hash_config(_jobs(pipeline)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "pipeline_sha256": sha256_file(ROOT / DEFAULT_PIPELINE),
        "no_scores_observed": True,
    }
    manifest = _output(pipeline) / "manifests/data_manifest.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text())
        if existing != payload:
            raise RuntimeError("a different compositional data manifest is already frozen")
        return existing
    atomic_json(payload, manifest)
    return payload


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _output(pipeline) / "manifests/data_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("run the freeze command before validation")
    manifest = json.loads(manifest_path.read_text())
    identities, tasks = _dataset_identities(pipeline)
    if identities != manifest["datasets"] or tasks != manifest["tasks"]:
        raise RuntimeError("compositional benchmark identity changed after freeze")
    if hash_config(_jobs(pipeline)) != manifest["roster_sha256"]:
        raise RuntimeError("model roster changed after compositional freeze")
    if sha256_file(PREREGISTRATION) != manifest["preregistration_sha256"]:
        raise RuntimeError("compositional preregistration changed after freeze")
    checkpoints = []
    for job in _jobs(pipeline):
        if job["kind"] == "student":
            path = ROOT / str(job["checkpoint"])
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append({"id": job["id"], "seed": job["seed"], "sha256": sha256_file(path)})
    payload = {
        "status": "READY",
        "tasks": tasks,
        "jobs": len(_jobs(pipeline)),
        "student_checkpoints": checkpoints,
        "no_optimizer_code_path": True,
    }
    atomic_json(payload, _output(pipeline) / "manifests/validation.json")
    return payload


def _decode_image(value: Any, parquet_root: Path) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    path = Path(str(value))
    if not path.is_absolute():
        path = parquet_root / path
    return Image.open(path).convert("RGB")


class SugarCrepeDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.rows, self.transform = rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {**row, "image": tensor}


class WinogroundDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Callable[[Image.Image], torch.Tensor], root: Path) -> None:
        self.rows, self.transform, self.root = rows, transform, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with _decode_image(row["image_0"], self.root) as image0:
            tensor0 = self.transform(image0)
        with _decode_image(row["image_1"], self.root) as image1:
            tensor1 = self.transform(image1)
        return {
            "sample_id": str(row.get("id", index)),
            "image_0": tensor0,
            "image_1": tensor1,
            "caption_0": str(row["caption_0"]),
            "caption_1": str(row["caption_1"]),
        }


def _collate_sugar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([row["image"] for row in rows]),
        "positive": [row["positive_caption"] for row in rows],
        "negative": [row["negative_caption"] for row in rows],
        "sample_id": [row["sample_id"] for row in rows],
        "category": [row["category"] for row in rows],
    }


def _collate_wino(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([value for row in rows for value in (row["image_0"], row["image_1"])]),
        "captions": [value for row in rows for value in (row["caption_0"], row["caption_1"])],
        "sample_id": [row["sample_id"] for row in rows],
        "caption_0": [row["caption_0"] for row in rows],
        "caption_1": [row["caption_1"] for row in rows],
    }


@torch.inference_mode()
def _evaluate_sugar(model: torch.nn.Module, transform: Callable, pipeline: dict[str, Any], device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    dataset = SugarCrepeDataset(_sugarcrepe_rows(pipeline), transform)
    loader = DataLoader(
        dataset,
        batch_size=int(pipeline["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        pin_memory=True,
        persistent_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) > 0,
        collate_fn=_collate_sugar,
    )
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = model.encode_image(batch["images"].to(device))
        positive = model.encode_text(batch["positive"])
        negative = model.encode_text(batch["negative"])
        positive_scores = torch.nn.functional.cosine_similarity(images.float(), positive.float())
        negative_scores = torch.nn.functional.cosine_similarity(images.float(), negative.float())
        for index, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "category": batch["category"][index],
                    "positive_score": float(positive_scores[index].cpu()),
                    "negative_score": float(negative_scores[index].cpu()),
                    "margin": float((positive_scores[index] - negative_scores[index]).cpu()),
                    "correct": bool(positive_scores[index] > negative_scores[index]),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = sugarcrepe_scores(
        torch.tensor(frame["positive_score"].to_numpy()),
        torch.tensor(frame["negative_score"].to_numpy()),
        frame["category"].tolist(),
    )
    return metrics, frame


@torch.inference_mode()
def _evaluate_winoground(model: torch.nn.Module, transform: Callable, pipeline: dict[str, Any], device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    path = ROOT / str(pipeline["datasets"]["winoground"]["parquet"])
    dataset = WinogroundDataset(_winoground_rows(path), transform, path.parent)
    loader = DataLoader(
        dataset,
        batch_size=int(pipeline["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        pin_memory=True,
        persistent_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) > 0,
        collate_fn=_collate_wino,
    )
    rows: list[dict[str, Any]] = []
    for batch in loader:
        # Reshape without assuming a model-specific embedding dimension.
        image = model.encode_image(batch["images"].to(device)).float()
        text = model.encode_text(batch["captions"]).float()
        dimension = image.shape[-1]
        image = image.view(-1, 2, dimension)
        text = text.view(-1, 2, dimension)
        score = torch.einsum("bid,bjd->bij", image, text)
        for index, sample_id in enumerate(batch["sample_id"]):
            s00, s01 = float(score[index, 0, 0].cpu()), float(score[index, 0, 1].cpu())
            s10, s11 = float(score[index, 1, 0].cpu()), float(score[index, 1, 1].cpu())
            text_ok = s00 > s01 and s11 > s10
            image_ok = s00 > s10 and s11 > s01
            rows.append(
                {
                    "sample_id": sample_id,
                    "caption_0": batch["caption_0"][index],
                    "caption_1": batch["caption_1"][index],
                    "s00": s00,
                    "s01": s01,
                    "s10": s10,
                    "s11": s11,
                    "text_correct": text_ok,
                    "image_correct": image_ok,
                    "group_correct": text_ok and image_ok,
                }
            )
    frame = pd.DataFrame(rows)
    score_tensor = torch.tensor(frame[["s00", "s01", "s10", "s11"]].to_numpy())
    return winoground_scores(score_tensor), frame


def evaluate(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    validation = validate(pipeline)
    manifest_path = _output(pipeline) / "manifests/data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else int(index)
    jobs = _jobs(pipeline)
    if not 0 <= selected < len(jobs):
        raise IndexError(f"evaluation index {selected} outside 0..{len(jobs)-1}")
    job = jobs[selected]
    run_name = f"seed_{job['seed']}" if job["seed"] is not None else "reference"
    destination = _output(pipeline) / "per_run" / job["id"] / run_name
    completed = destination / "evaluation.json"
    if completed.is_file():
        existing = json.loads(completed.read_text())
        if existing.get("manifest_sha256") == sha256_file(manifest_path):
            return {**existing, "status": "RESUMED"}
        raise RuntimeError(f"stale compositional result exists: {completed}")
    device, hardware = _hardware_guard(pipeline)
    if job["kind"] == "student":
        model, config, provenance = _load_student(job, device)
        transform = _student_transform(config)
    else:
        model = build_reference(job["id"]).to(device).eval()
        transform = model.preprocess
        provenance = {
            **parameter_counts_reference(model),
            "inference_trainable_parameters": 0,
            "reference_weights": _reference_weight_manifest(model, job["id"]),
        }
    if any(module.training for module in model.modules()):
        raise RuntimeError("evaluation model contains a module in training mode")
    destination.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if "sugarcrepe" in manifest["tasks"]:
            sugar_metrics, sugar_predictions = _evaluate_sugar(model, transform, pipeline, device)
            atomic_csv(sugar_predictions, destination / "sugarcrepe_predictions.csv")
            atomic_json(sugar_metrics, destination / "sugarcrepe_metrics.json")
            metrics["sugarcrepe"] = sugar_metrics
        if "winoground" in manifest["tasks"]:
            wino_metrics, wino_predictions = _evaluate_winoground(model, transform, pipeline, device)
            atomic_csv(wino_predictions, destination / "winoground_predictions.csv")
            atomic_json(wino_metrics, destination / "winoground_metrics.json")
            metrics["winoground"] = wino_metrics
    payload = {
        "status": "COMPLETE",
        **job,
        **hardware,
        **provenance,
        "metrics": metrics,
        "tasks": manifest["tasks"],
        "manifest_sha256": sha256_file(manifest_path),
        "validation_status": validation["status"],
        "no_training_performed": True,
    }
    atomic_json(payload, completed)
    return payload


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((_output(pipeline) / "manifests/data_manifest.json").read_text())
    evaluations = []
    for job in _jobs(pipeline):
        run_name = f"seed_{job['seed']}" if job["seed"] is not None else "reference"
        path = _output(pipeline) / "per_run" / job["id"] / run_name / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing compositional evaluation: {path}")
        evaluations.append(json.loads(path.read_text()))
    rows: list[dict[str, Any]] = []
    for value in evaluations:
        for dataset, metrics in value["metrics"].items():
            for metric, score in metrics.items():
                rows.append(
                    {
                        "model_id": value["id"],
                        "label": value["label"],
                        "kind": value["kind"],
                        "seed": value["seed"],
                        "dataset": dataset,
                        "metric": metric,
                        "value": float(score),
                    }
                )
    frame = pd.DataFrame(rows)
    aggregate_rows = []
    for keys, group in frame.groupby(["model_id", "label", "kind", "dataset", "metric"], sort=False):
        mean, sd = _mean_sd(group["value"].tolist())
        aggregate_rows.append(
            dict(zip(["model_id", "label", "kind", "dataset", "metric"], keys))
            | {"n": len(group), "mean": mean, "sd": sd, "min": float(group["value"].min()), "max": float(group["value"].max())}
        )
    aggregate = pd.DataFrame(aggregate_rows)
    root = _output(pipeline) / "report"
    atomic_csv(frame, root / "per_seed_results.csv")
    atomic_csv(aggregate, root / "aggregate_results.csv")
    primary = aggregate[
        aggregate["dataset"].eq(pipeline["evaluation"]["primary_dataset"])
        & aggregate["metric"].eq(pipeline["evaluation"]["primary_metric"])
    ]
    lines = [
        "# Untouched compositional evaluation",
        "",
        "No learning, prompt tuning, checkpoint selection or threshold tuning was performed.",
        "",
        "## Primary: SugarCrepe overall accuracy",
        "",
        primary.to_markdown(index=False),
        "",
        "## Complete diagnostics",
        "",
        aggregate.to_markdown(index=False),
    ]
    atomic_text("\n".join(lines) + "\n", root / "report.md")
    payload = {"status": "COMPLETE", "tasks": manifest["tasks"], "evaluations": len(evaluations), "aggregate_rows": aggregate_rows}
    atomic_json(payload, root / "report.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("acquire-sugarcrepe", "acquire-winoground", "freeze", "validate", "evaluate", "report"),
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    parser.add_argument("--acknowledge-winoground-license", action="store_true")
    args = parser.parse_args()
    pipeline = _pipeline(args.pipeline)
    if args.command == "acquire-sugarcrepe":
        result = acquire_sugarcrepe(pipeline)
    elif args.command == "acquire-winoground":
        result = acquire_winoground(pipeline, acknowledge_license=args.acknowledge_winoground_license)
    elif args.command == "freeze":
        result = freeze(pipeline)
    elif args.command == "validate":
        result = validate(pipeline)
    elif args.command == "evaluate":
        result = evaluate(pipeline, args.index)
    else:
        result = report(pipeline)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
