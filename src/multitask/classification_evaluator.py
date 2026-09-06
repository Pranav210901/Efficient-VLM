from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.transforms import build_image_transform

from .config import PROMPT_TEMPLATES
from .datasets import load_zeroshot_dataset, zeroshot_subset_indices
from .prompt_templates import clean_class_name, ensemble_class_embeddings


class IndexedClassificationDataset(Dataset):
    """Add stable sample IDs to a torchvision classification dataset."""

    def __init__(self, dataset: Dataset, task: str, project_root: str | Path) -> None:
        self.dataset = dataset
        self.task = task
        self.project_root = Path(project_root).resolve()

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_id(self, index: int) -> str:
        value: str | Path | None = None
        samples = getattr(self.dataset, "samples", None)
        images = getattr(self.dataset, "_images", None)
        if samples is not None:
            value = samples[index][0]
        elif images is not None:
            value = images[index]
        if value is None:
            return f"{self.task}:{index:06d}"
        path = Path(value)
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, int(target), self._sample_id(index)


def prepare_zeroshot_dataset(
    project_root: str | Path,
    task: str,
    image_size: int,
    split_role: str = "historical",
) -> tuple[Dataset, list[str]]:
    """Build an evaluation-only dataset and natural-language class labels."""
    transform = build_image_transform(image_size=image_size, train=False)
    dataset = load_zeroshot_dataset(project_root, task, transform=transform, split_role=split_role)
    class_names = [clean_class_name(str(value)) for value in dataset.classes]
    wrapped = IndexedClassificationDataset(dataset, task, project_root)
    indices = zeroshot_subset_indices(project_root, task, dataset, split_role)
    return (Subset(wrapped, indices) if indices is not None else wrapped), class_names


def _validate_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int | None = None) -> int:
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.numel():
        raise ValueError("logits must be [samples, classes] and targets must be [samples]")
    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError("classification evaluation requires at least one sample and class")
    classes = logits.size(1) if num_classes is None else num_classes
    if classes != logits.size(1):
        raise ValueError("num_classes must equal logits.shape[1]")
    if targets.min().item() < 0 or targets.max().item() >= classes:
        raise ValueError("targets contain a class index outside the logits columns")
    return classes


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute task-level zero-shot metrics from unmodified model logits."""
    classes = _validate_logits(logits, targets, num_classes)
    predictions = logits.argmax(dim=1)
    topk = logits.topk(min(5, classes), dim=1).indices
    recalls, f1s = [], []
    for label in range(classes):
        truth = targets.eq(label)
        predicted = predictions.eq(label)
        tp = (truth & predicted).sum().item()
        fp = (~truth & predicted).sum().item()
        fn = (truth & ~predicted).sum().item()
        recalls.append(tp / max(1, tp + fn))
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))

    probabilities = logits.float().softmax(dim=1)
    confidence = probabilities.max(dim=1).values
    best_two = probabilities.topk(min(2, classes), dim=1).values
    top1_margin = best_two[:, 0] if classes == 1 else best_two[:, 0] - best_two[:, 1]
    correct_score = probabilities.gather(1, targets[:, None]).squeeze(1)
    if classes == 1:
        true_class_margin = correct_score
    else:
        incorrect = probabilities.clone()
        incorrect.scatter_(1, targets[:, None], float("-inf"))
        true_class_margin = correct_score - incorrect.max(dim=1).values

    return {
        "top1_accuracy": float(predictions.eq(targets).float().mean()),
        "top5_accuracy": float(topk.eq(targets[:, None]).any(dim=1).float().mean()),
        "macro_f1": float(sum(f1s) / len(f1s)),
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "mean_confidence": float(confidence.mean()),
        "classification_margin": float(true_class_margin.mean()),
        "top1_margin": float(top1_margin.mean()),
    }


def per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, object]]:
    _validate_logits(logits, targets, len(class_names))
    predictions = logits.argmax(dim=1)
    rows = []
    for index, name in enumerate(class_names):
        selected = targets.eq(index)
        rows.append(
            {
                "class_index": index,
                "class": name,
                "accuracy": float(predictions[selected].eq(index).float().mean()) if selected.any() else float("nan"),
                "samples": int(selected.sum()),
            }
        )
    return rows


def classification_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_ids: list[str],
    class_names: list[str],
    task: str,
) -> list[dict[str, Any]]:
    """Create cache-safe sample rows for error and complementarity analysis."""
    classes = _validate_logits(logits, targets, len(class_names))
    if len(sample_ids) != targets.numel():
        raise ValueError("sample_ids must contain one value per target")

    probabilities = logits.float().softmax(dim=1).cpu()
    targets = targets.cpu()
    top_count = min(5, classes)
    top_scores, top_indices = probabilities.topk(top_count, dim=1)
    predictions = top_indices[:, 0]
    if classes > 1:
        incorrect = probabilities.clone()
        incorrect.scatter_(1, targets[:, None], float("-inf"))
        incorrect_scores, incorrect_indices = incorrect.max(dim=1)
    else:
        incorrect_scores = torch.zeros_like(probabilities[:, 0])
        incorrect_indices = torch.zeros_like(targets)

    rows = []
    for index, sample_id in enumerate(sample_ids):
        target = int(targets[index])
        prediction = int(predictions[index])
        correct_score = float(probabilities[index, target])
        strongest_incorrect = float(incorrect_scores[index])
        top1_margin = float(top_scores[index, 0])
        if top_count > 1:
            top1_margin -= float(top_scores[index, 1])
        rows.append(
            {
                "task": task,
                "sample_id": str(sample_id),
                "target_index": target,
                "target": class_names[target],
                "prediction_index": prediction,
                "prediction": class_names[prediction],
                "correct": prediction == target,
                "confidence": float(top_scores[index, 0]),
                "correct_class_score": correct_score,
                "highest_incorrect_index": int(incorrect_indices[index]),
                "highest_incorrect": class_names[int(incorrect_indices[index])],
                "highest_incorrect_score": strongest_incorrect,
                "classification_margin": correct_score - strongest_incorrect,
                "top1_margin": top1_margin,
                "top5": [
                    {
                        "class_index": int(class_index),
                        "class": class_names[int(class_index)],
                        "score": float(score),
                    }
                    for class_index, score in zip(top_indices[index], top_scores[index])
                ],
            }
        )
    return rows


@torch.inference_mode()
def evaluate_zeroshot_classification(
    model,
    dataset: Dataset,
    class_names: list[str],
    device: str | torch.device,
    task: str,
    batch_size: int = 64,
    num_workers: int = 0,
    max_samples: int | None = None,
    templates: tuple[str, ...] = PROMPT_TEMPLATES,
) -> dict[str, Any]:
    """Evaluate frozen image/text projections as a prompt-based classifier."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when supplied")
    selected: Dataset = dataset
    if max_samples is not None and max_samples < len(dataset):
        selected = Subset(dataset, range(max_samples))
    if len(selected) == 0:
        raise ValueError("classification dataset is empty")

    device = torch.device(device)
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )
    model.eval()
    class_embeddings = ensemble_class_embeddings(model, class_names, templates=templates).to(device)
    class_embeddings = F.normalize(class_embeddings, dim=-1)
    scale = getattr(model, "logit_scale", None)
    logit_scale = scale.exp().clamp(max=100.0) if scale is not None else torch.tensor(1.0, device=device)

    print(
        f"Evaluating {task}: {len(selected)} images, {len(class_names)} classes, {len(loader)} batches",
        flush=True,
    )
    logits_parts, target_parts, sample_ids = [], [], []
    interval = max(1, len(loader) // 10)
    for batch_index, (images, targets, identifiers) in enumerate(loader, 1):
        image_embeddings = F.normalize(model.encode_image(images.to(device, non_blocking=True)), dim=-1)
        logits_parts.append((logit_scale * image_embeddings @ class_embeddings.t()).float().cpu())
        target_parts.append(targets.long().cpu())
        sample_ids.extend(str(value) for value in identifiers)
        if batch_index == 1 or batch_index == len(loader) or batch_index % interval == 0:
            print(f"{task}: {batch_index}/{len(loader)} batches", flush=True)

    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"Non-finite zero-shot logits for {task}")
    return {
        "logits": logits,
        "targets": targets,
        "sample_ids": sample_ids,
        "class_names": class_names,
        "metrics": classification_metrics(logits, targets, len(class_names)),
        "per_class": per_class_accuracy(logits, targets, class_names),
    }
