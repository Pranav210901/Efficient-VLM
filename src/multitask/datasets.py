from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .task_registry import TASK_REGISTRY


ZERO_SHOT_DATASET_PATHS = {
    "cifar100_zeroshot": Path("data/multitask/cifar100"),
    "pets_zeroshot": Path("data/multitask/oxford-iiit-pet"),
    "eurosat_zeroshot": Path("data/multitask/eurosat"),
}


def load_zeroshot_dataset(
    project_root: str | Path,
    task: str,
    transform: Callable[[Any], Any] | None = None,
    split_role: str = "historical",
):
    # open a supported public zero-shot dataset without network access
    if split_role not in {"historical", "development", "final_test"}:
        raise ValueError("split_role must be historical, development, or final_test")
    try:
        from torchvision.datasets import CIFAR100, EuroSAT, OxfordIIITPet
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("torchvision is required for zero-shot classification evaluation") from exc

    root = Path(project_root)
    if task == "cifar100_zeroshot":
        return CIFAR100(
            root=root / ZERO_SHOT_DATASET_PATHS[task],
            train=split_role == "development",
            transform=transform,
            download=False,
        )
    if task == "pets_zeroshot":
        return OxfordIIITPet(
            root=root / ZERO_SHOT_DATASET_PATHS[task],
            split="trainval" if split_role == "development" else "test",
            target_types="category",
            transform=transform,
            download=False,
        )
    if task == "eurosat_zeroshot":
        # torchvision exposes the complete EuroSAT archive; it has no official
        # train/test split. The model is never trained on EuroSAT in this workflow.
        return EuroSAT(
            root=root / ZERO_SHOT_DATASET_PATHS[task],
            transform=transform,
            download=False,
        )
    raise KeyError(f"Unsupported zero-shot dataset {task!r}; choose from {sorted(ZERO_SHOT_DATASET_PATHS)}")


def zeroshot_split_name(task: str, split_role: str = "historical") -> str:
    if split_role == "development":
        return {
            "cifar100_zeroshot": "deterministic development partition of official train",
            "pets_zeroshot": "deterministic development partition of trainval",
            "eurosat_zeroshot": "stored deterministic stratified development partition",
        }[task]
    if split_role == "final_test":
        return {
            "cifar100_zeroshot": "reserved official test split",
            "pets_zeroshot": "reserved official test split",
            "eurosat_zeroshot": "stored deterministic stratified test partition",
        }[task]
    return {
        "cifar100_zeroshot": "test split",
        "pets_zeroshot": "test split",
        "eurosat_zeroshot": "complete dataset (torchvision provides no official split)",
    }[task]


def dataset_split_id(task: str, split_role: str) -> str:
    if task == "coco_retrieval":
        return "coco_val2017_development" if split_role != "final_test" else "external_retrieval_final_test_unavailable"
    values = {
        "historical": {
            "cifar100_zeroshot": "cifar100_official_test_exploratory",
            "pets_zeroshot": "oxford_iiit_pet_official_test_exploratory",
            "eurosat_zeroshot": "eurosat_complete_dataset_exploratory",
        },
        "development": {
            "cifar100_zeroshot": "cifar100_official_train_development",
            "pets_zeroshot": "oxford_iiit_pet_trainval_development",
            "eurosat_zeroshot": "eurosat_deterministic_stratified_development",
        },
        "final_test": {
            "cifar100_zeroshot": "cifar100_official_test_reserved",
            "pets_zeroshot": "oxford_iiit_pet_official_test_reserved",
            "eurosat_zeroshot": "eurosat_deterministic_stratified_test_reserved",
        },
    }
    return values[split_role][task]


def zeroshot_subset_indices(project_root: str | Path, task: str, dataset: Any, split_role: str) -> list[int] | None:
    # return fixed split indices, or ``None`` when the full loaded dataset applies
    if split_role == "historical" or (split_role == "final_test" and task != "eurosat_zeroshot"):
        return None
    if split_role == "development" and task in {"cifar100_zeroshot", "pets_zeroshot"}:
        labels = list(getattr(dataset, "targets", getattr(dataset, "_labels", [])))
        if not labels:
            raise ValueError(f"Cannot locate labels for deterministic {task} development split")
        rng = np.random.RandomState(42)
        values = np.asarray(labels)
        selected: list[int] = []
        for label in sorted(set(int(value) for value in labels)):
            indices = np.flatnonzero(values == label)
            indices = indices[rng.permutation(len(indices))]
            train_end = int(round(len(indices) * 0.80))
            selected.extend(int(value) for value in indices[train_end:])
        return sorted(selected)
    if task == "eurosat_zeroshot" and split_role in {"development", "final_test"}:
        from src.phase15.evaluation_protocol import create_eurosat_split_manifest

        import pandas as pd

        root = Path(project_root).resolve()
        manifest = pd.read_csv(create_eurosat_split_manifest(root))
        desired = "development" if split_role == "development" else "test"
        allowed = set(manifest.loc[manifest["split"].eq(desired), "sample_id"].astype(str))
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise ValueError("EuroSAT dataset has no stable sample paths")
        indices = []
        for index, value in enumerate(samples):
            path = Path(value[0]).resolve()
            try:
                sample_id = str(path.relative_to(root))
            except ValueError:
                sample_id = str(path)
            if sample_id in allowed:
                indices.append(index)
        if len(indices) != len(allowed):
            raise ValueError(f"EuroSAT {desired} manifest coverage mismatch: {len(indices)} != {len(allowed)}")
        return indices
    return None


def _count_csv(path: Path) -> tuple[int, int | None]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = {row.get("label") or row.get("class") for row in rows} - {None, ""}
    return len(rows), len(classes) or None


def validate_datasets(project_root: str | Path, datasets: list[str] | tuple[str, ...], split_role: str = "historical") -> list[dict[str, object]]:
    root = Path(project_root)
    locations = {
        "coco_retrieval": root / "data/val_all_captions.csv",
        **{name: root / path for name, path in ZERO_SHOT_DATASET_PATHS.items()},
        "imagenet1k_zeroshot": root / "data/multitask/imagenet",
        "winoground": root / "data/multitask/winoground",
        "sugarcrepe": root / "data/multitask/sugarcrepe",
    }
    results = []
    for name in datasets:
        spec = TASK_REGISTRY[name]
        path = locations[name]
        status, samples, classes, detail = "missing", 0, None, ""
        if name == "coco_retrieval" and path.is_file():
            samples, _ = _count_csv(path)
            status = "ready" if samples else "invalid"
            detail = "standard five-caption CSV"
        elif name in ZERO_SHOT_DATASET_PATHS:
            try:
                dataset = load_zeroshot_dataset(root, name, split_role=split_role)
                indices = zeroshot_subset_indices(root, name, dataset, split_role)
                samples = len(indices) if indices is not None else len(dataset)
                classes = len(dataset.classes)
                status = "ready" if samples > 0 and classes > 1 else "invalid"
                detail = f"torchvision {zeroshot_split_name(name, split_role)}; download disabled"
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                status = "missing" if not path.exists() else "invalid"
                detail = str(exc).splitlines()[0]
        elif path.exists():
            files = [value for value in path.rglob("*") if value.is_file()]
            samples = len(files)
            status = "ready" if files else "invalid"
            if name == "winoground":
                candidates = list(path.rglob("*.jsonl")) + list(path.rglob("*.json"))
                detail = "metadata found" if candidates else "missing JSON metadata"
                status = status if candidates else "invalid"
            elif name == "sugarcrepe":
                candidates = list(path.rglob("*.json")) + list(path.rglob("*.csv"))
                detail = "annotations found" if candidates else "missing annotations"
                status = status if candidates else "invalid"
            else:
                class_dirs = [value for value in path.rglob("*") if value.is_dir() and any(child.is_file() for child in value.iterdir())]
                classes = len(class_dirs) or None
        optional = not spec.mandatory
        if status == "missing":
            if name == "imagenet1k_zeroshot":
                detail = "optional; place ImageNet under data/multitask/imagenet (never auto-downloaded)"
            elif spec.group == "classification":
                detail = f"download once with torchvision, then place it under {path.relative_to(root)}"
            else:
                detail = f"place benchmark files under {path.relative_to(root)}; licensing may require manual access"
        results.append({"task": name, "group": spec.group, "status": status, "samples": samples, "classes_or_categories": classes, "optional": optional, "path": str(path), "detail": detail, "split_role": split_role, "dataset_split": dataset_split_id(name, split_role) if name in {*ZERO_SHOT_DATASET_PATHS, "coco_retrieval"} else "evaluation_only"})
    return results


def validate_coco(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    required = {
        "train_csv": root / "data/train.csv", "val_csv": root / "data/val.csv",
        "train_images": root / "data/coco/train2017", "val_images": root / "data/coco/val2017",
        "train_annotations": root / "data/coco/annotations/captions_train2017.json",
        "val_annotations": root / "data/coco/annotations/captions_val2017.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    report: dict[str, object] = {"missing": missing, "ready": not missing}
    for split in ("train", "val"):
        path = required[f"{split}_csv"]
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        report[f"{split}_rows"] = len(rows)
        report[f"{split}_images"] = len({row.get("image_path") for row in rows})
        report[f"{split}_duplicates"] = len(rows) - len({(row.get("image_path"), row.get("caption")) for row in rows})
        report[f"{split}_nulls"] = sum(not row.get("image_path") or not row.get("caption") for row in rows)
    if required["val_annotations"].exists():
        report["validation_captions"] = len(json.loads(required["val_annotations"].read_text()).get("annotations", []))
    if required["train_csv"].exists() and required["val_csv"].exists():
        def ids(path: Path) -> set[str]:
            with path.open(newline="") as handle:
                return {str(row.get("image_path")) for row in csv.DictReader(handle)}
        report["train_validation_overlap"] = len(ids(required["train_csv"]) & ids(required["val_csv"]))
    return report
