"""Frozen development/final-test policy and deterministic split manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io_utils import atomic_csv, atomic_json, atomic_text


def load_evaluation_protocol(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root).resolve() / "configs/evaluation_protocol.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation protocol is missing: {path}")
    payload = yaml.safe_load(path.read_text())
    required = {"schema_version", "protocol_status", "fixed_seed", "datasets", "selection_weights", "diversity_constraints"}
    missing = required - set(payload or {})
    if missing:
        raise ValueError(f"Evaluation protocol is missing keys: {sorted(missing)}")
    return payload


def stratified_split_manifest(
    sample_ids: list[str],
    class_ids: list[int],
    class_names: list[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    development_fraction: float = 0.15,
) -> pd.DataFrame:
    if not (len(sample_ids) == len(class_ids) == len(class_names)):
        raise ValueError("sample_ids, class_ids, and class_names must have equal length")
    if not 0 < train_fraction < 1 or not 0 < development_fraction < 1 or train_fraction + development_fraction >= 1:
        raise ValueError("split fractions must be positive and leave a non-empty test fraction")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    labels = np.asarray(class_ids)
    for class_id in sorted(set(class_ids)):
        indices = np.flatnonzero(labels == class_id)
        indices = indices[rng.permutation(len(indices))]
        train_end = int(round(len(indices) * train_fraction))
        development_end = train_end + int(round(len(indices) * development_fraction))
        development_end = min(development_end, len(indices) - 1) if len(indices) >= 3 else development_end
        for position, index in enumerate(indices):
            split = "train" if position < train_end else "development" if position < development_end else "test"
            rows.append(
                {
                    "sample_id": sample_ids[int(index)],
                    "class_id": int(class_ids[int(index)]),
                    "class_name": class_names[int(index)],
                    "split": split,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows).sort_values("sample_id", kind="stable").reset_index(drop=True)


def create_eurosat_split_manifest(project_root: str | Path, force: bool = False) -> Path:
    root = Path(project_root).resolve()
    destination = root / "data/splits/eurosat_split_manifest.csv"
    if destination.exists() and not force:
        existing = pd.read_csv(destination)
        required = {"sample_id", "class_id", "class_name", "split", "seed"}
        if required.issubset(existing.columns) and not existing["sample_id"].duplicated().any():
            return destination
        raise ValueError(f"Existing EuroSAT split manifest is invalid: {destination}")
    image_root = root / "data/multitask/eurosat/eurosat/2750"
    if not image_root.exists():
        raise FileNotFoundError(f"EuroSAT images are unavailable: {image_root}")
    classes = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    class_to_id = {name: index for index, name in enumerate(classes)}
    samples: list[tuple[str, int, str]] = []
    for class_name in classes:
        for path in sorted((image_root / class_name).glob("*")):
            if path.is_file():
                samples.append((str(path.relative_to(root)), class_to_id[class_name], class_name))
    protocol = load_evaluation_protocol(root)
    split = protocol.get("split_policy", {})
    frame = stratified_split_manifest(
        [value[0] for value in samples],
        [value[1] for value in samples],
        [value[2] for value in samples],
        seed=int(protocol["fixed_seed"]),
        train_fraction=float(split.get("eurosat_train_fraction", 0.70)),
        development_fraction=float(split.get("eurosat_development_fraction", 0.15)),
    )
    atomic_csv(frame, destination)
    return destination


def write_evaluation_protocol_outputs(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    protocol = load_evaluation_protocol(root)
    eurosat_manifest: str | None = None
    try:
        eurosat_manifest = str(create_eurosat_split_manifest(root))
    except FileNotFoundError:
        eurosat_manifest = None
    output = root / "results/phase15/methodology"
    json_path = output / "evaluation_protocol.json"
    usage_path = output / "data_usage_table.csv"
    audit_path = output / "leakage_audit.md"
    payload = dict(protocol)
    payload["source"] = str(root / "configs/evaluation_protocol.yaml")
    payload["eurosat_split_manifest"] = eurosat_manifest
    atomic_json(payload, json_path)
    usage = pd.DataFrame(protocol["datasets"])
    atomic_csv(usage, usage_path)
    external_available = (root / "data/multitask/flickr30k").exists()
    markdown = [
        "# Evaluation and leakage audit",
        "",
        "The protocol was frozen before Phase 2. Existing CIFAR-100 test, Pets test, and complete-EuroSAT results are retained as exploratory/development evidence; they are not untouched confirmatory results.",
        "",
        "## Going-forward policy",
        "",
        "- COCO val2017 is development and expert-selection evidence. It is not described as an untouched final test set.",
        "- CIFAR-100 selection uses a deterministic subset of the official training split; official test is reserved for final reporting.",
        "- Oxford-IIIT Pets selection uses a deterministic partition of trainval; official test is reserved for final reporting.",
        "- EuroSAT uses the fixed stratified manifest under data/splits; its stored test partition is reserved for final reporting.",
        "- Winoground and SugarCrepe are evaluation-only and cannot supply training negatives.",
        "",
        "## Untouched external validation",
        "",
        f"Flickr30k external retrieval data currently available: **{'yes' if external_available else 'no'}**. Until an external benchmark is installed, no fully untouched external retrieval result is claimed.",
        "",
        "## Data-usage table",
        "",
        usage.to_markdown(index=False),
    ]
    atomic_text("\n".join(markdown) + "\n", audit_path)
    return {"json": str(json_path), "data_usage": str(usage_path), "leakage_audit": str(audit_path), "eurosat_manifest": eurosat_manifest or "missing"}


def protected_test_ids(project_root: str | Path) -> set[str]:
    root = Path(project_root).resolve()
    protected = {f"cifar100_zeroshot:{index:06d}" for index in range(10000)}
    pets_images = root / "data/multitask/oxford-iiit-pet/oxford-iiit-pet/images"
    test_list = root / "data/multitask/oxford-iiit-pet/oxford-iiit-pet/annotations/test.txt"
    if test_list.exists():
        for line in test_list.read_text().splitlines():
            name = line.split()[0] if line.strip() else ""
            if name:
                protected.add(str((pets_images / f"{name}.jpg").relative_to(root)))
    manifest = root / "data/splits/eurosat_split_manifest.csv"
    if manifest.exists():
        values = pd.read_csv(manifest)
        protected.update(values.loc[values["split"].eq("test"), "sample_id"].astype(str))
    return protected
