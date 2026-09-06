from __future__ import annotations

import csv
import json
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_image_transform

from .checkpoint_selection import CheckpointSpec, load_frozen_model
from .embedding_cache import load_cache, save_cache
from .retrieval_evaluator import text_to_image_predictions


@dataclass(frozen=True)
class QualitativeModel:
    label: str
    vision_encoder: str
    text_encoder: str
    variant: str
    checkpoint: str | Path

    def checkpoint_spec(self, project_root: Path) -> CheckpointSpec:
        path = Path(self.checkpoint)
        path = path if path.is_absolute() else project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Qualitative retrieval checkpoint is missing: {path}")
        return CheckpointSpec(self.vision_encoder, self.text_encoder, self.variant, path, 0)


class _GalleryImages(Dataset):
    def __init__(self, paths: Sequence[Path], image_size: int) -> None:
        self.paths = list(paths)
        self.transform = build_image_transform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _read_coco_gallery(
    csv_path: Path,
    project_root: Path,
    max_images: int | None = None,
) -> tuple[list[str], list[Path], list[list[str]]]:
    grouped: dict[str, list[str]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["image_path"]), []).append(str(row["caption"]))
    items = list(grouped.items())[:max_images]
    if not items:
        raise ValueError(f"No image-caption rows found in {csv_path}")
    image_ids = [value for value, _ in items]
    paths = [Path(value) if Path(value).is_absolute() else project_root / value for value in image_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"COCO gallery image is missing: {missing[0]}")
    return image_ids, paths, [captions for _, captions in items]


@torch.inference_mode()
def _encode_gallery(
    model,
    paths: Sequence[Path],
    captions: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        _GalleryImages(paths, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )
    image_parts = []
    interval = max(1, len(loader) // 10)
    print(f"{label}: encoding {len(paths)} gallery images in {len(loader)} batches", flush=True)
    for batch_index, images in enumerate(loader, 1):
        image_parts.append(F.normalize(model.encode_image(images.to(device, non_blocking=True)), dim=-1).cpu())
        if batch_index == 1 or batch_index == len(loader) or batch_index % interval == 0:
            print(f"{label}: gallery batch {batch_index}/{len(loader)}", flush=True)
    text_embeddings = F.normalize(model.encode_text(list(captions)), dim=-1).cpu()
    return torch.cat(image_parts), text_embeddings


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_qualitative_retrieval(
    project_root: str | Path,
    models: Sequence[QualitativeModel],
    device: str | torch.device | None = None,
    num_queries: int = 3,
    topk: int = 5,
    query_seed: int = 42,
    max_images: int | None = None,
    output_dir: str | Path = "results/qualitative_retrieval",
) -> dict[str, Any]:
    """Compare text-to-image retrieval on one deterministic COCO gallery."""
    if len(models) < 2:
        raise ValueError("At least two models are required for a qualitative comparison")
    if num_queries < 1 or topk < 1:
        raise ValueError("num_queries and topk must be positive")

    root = Path(project_root).resolve()
    csv_path = root / "data/val_all_captions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Held-out COCO caption file is missing: {csv_path}")
    output = Path(output_dir)
    output = output if output.is_absolute() else root / output
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    image_ids, image_paths, caption_groups = _read_coco_gallery(csv_path, root, max_images=max_images)
    if num_queries > len(image_ids):
        raise ValueError("num_queries cannot exceed the number of gallery images")
    query_gallery_indices = sorted(random.Random(query_seed).sample(range(len(image_ids)), num_queries))
    query_captions = [caption_groups[index][0] for index in query_gallery_indices]
    query_target_ids = [image_ids[index] for index in query_gallery_indices]

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_info in models:
        spec = model_info.checkpoint_spec(root)
        cache_path = cache_dir / f"{_slug(model_info.label)}.pt"
        metadata = {
            "evaluator_version": 1,
            "checkpoint": str(spec.checkpoint),
            "checkpoint_mtime_ns": spec.checkpoint.stat().st_mtime_ns,
            "dataset": str(csv_path),
            "dataset_mtime_ns": csv_path.stat().st_mtime_ns,
            "gallery_images": len(image_ids),
            "max_images": max_images,
            "query_seed": query_seed,
            "query_gallery_indices": query_gallery_indices,
            "query_captions": query_captions,
        }
        cached = load_cache(cache_path, metadata)
        if cached is None:
            print(f"Loading qualitative model on {device}: {model_info.label}", flush=True)
            model, config, best_epoch = load_frozen_model(spec, str(device))
            image_size = int(config.get("data", {}).get("image_size", 224))
            batch_size = int(config.get("training", {}).get("batch_size", 64))
            num_workers = min(4, int(config.get("data", {}).get("num_workers", 0)))
            image_embeddings, text_embeddings = _encode_gallery(
                model,
                image_paths,
                query_captions,
                image_size,
                batch_size,
                num_workers,
                device,
                model_info.label,
            )
            cached = {
                "image_embeddings": image_embeddings,
                "text_embeddings": text_embeddings,
                "image_ids": image_ids,
                "query_target_ids": query_target_ids,
                "best_epoch": best_epoch,
            }
            save_cache(cache_path, cached, metadata)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Saved qualitative retrieval cache: {cache_path}", flush=True)
        else:
            print(f"Reusing qualitative retrieval cache: {cache_path}", flush=True)

        predictions = text_to_image_predictions(
            cached["image_embeddings"],
            cached["text_embeddings"],
            list(cached["image_ids"]),
            list(cached["query_target_ids"]),
            topk=topk,
        )
        for query_number, prediction in enumerate(predictions, 1):
            candidates = [
                {
                    "rank": rank,
                    "image_path": image_id,
                    "score": score,
                    "is_target": image_id == prediction["target_image_id"],
                }
                for rank, (image_id, score) in enumerate(
                    zip(prediction["top_candidate_ids"], prediction["top_candidate_scores"]),
                    1,
                )
            ]
            all_rows.append(
                {
                    "model": model_info.label,
                    "checkpoint": str(spec.checkpoint),
                    "best_epoch": int(cached.get("best_epoch", 0)),
                    "query_number": query_number,
                    "query_seed": query_seed,
                    "caption": query_captions[query_number - 1],
                    "target_image": prediction["target_image_id"],
                    "correct_target_rank": prediction["correct_target_rank"],
                    "recall_at_1": prediction["recall_at_1"],
                    "recall_at_5": prediction["recall_at_5"],
                    "recall_at_10": prediction["recall_at_10"],
                    "top_images": candidates,
                }
            )
        model_rows = all_rows[-len(predictions) :]
        summary_rows.append(
            {
                "model": model_info.label,
                "checkpoint": str(spec.checkpoint),
                "gallery_images": len(image_ids),
                "qualitative_queries": len(model_rows),
                "R@1": sum(row["recall_at_1"] for row in model_rows) / len(model_rows),
                "R@5": sum(row["recall_at_5"] for row in model_rows) / len(model_rows),
                "R@10": sum(row["recall_at_10"] for row in model_rows) / len(model_rows),
                "mean_target_rank": sum(row["correct_target_rank"] for row in model_rows) / len(model_rows),
            }
        )

    examples_path = output / "retrieval_examples.jsonl"
    examples_path.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    summary_path = output / "qualitative_summary.csv"
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    return {
        "rows": all_rows,
        "summary": summary,
        "examples_path": examples_path,
        "summary_path": summary_path,
        "query_indices": query_gallery_indices,
        "query_seed": query_seed,
    }


def load_qualitative_retrieval(output_dir: str | Path) -> dict[str, Any] | None:
    output = Path(output_dir)
    examples_path = output / "retrieval_examples.jsonl"
    summary_path = output / "qualitative_summary.csv"
    if not examples_path.is_file() or not summary_path.is_file():
        return None
    rows = [json.loads(line) for line in examples_path.read_text().splitlines() if line.strip()]
    return {
        "rows": rows,
        "summary": pd.read_csv(summary_path),
        "examples_path": examples_path,
        "summary_path": summary_path,
    }


def display_qualitative_retrieval(
    rows: Sequence[dict[str, Any]],
    project_root: str | Path,
) -> None:
    """Display one target-plus-retrieval montage for each shared caption."""
    import matplotlib.pyplot as plt

    root = Path(project_root).resolve()
    query_numbers = list(dict.fromkeys(int(row["query_number"]) for row in rows))
    for query_number in query_numbers:
        query_rows = [row for row in rows if int(row["query_number"]) == query_number]
        if not query_rows:
            continue
        topk = max(len(row["top_images"]) for row in query_rows)
        figure, axes = plt.subplots(len(query_rows), topk + 1, figsize=(3.0 * (topk + 1), 3.3 * len(query_rows)), squeeze=False)
        caption = str(query_rows[0]["caption"])
        figure.suptitle(f'Query {query_number}: "{textwrap.fill(caption, 90)}"', fontsize=13)
        for row_index, row in enumerate(query_rows):
            target = Path(row["target_image"])
            target = target if target.is_absolute() else root / target
            with Image.open(target) as image:
                axes[row_index, 0].imshow(image.convert("RGB"))
            axes[row_index, 0].set_title("Ground truth")
            axes[row_index, 0].set_ylabel(
                f"{row['model']}\nGT rank: {row['correct_target_rank']}",
                rotation=0,
                ha="right",
                va="center",
                labelpad=90,
            )
            axes[row_index, 0].axis("off")
            for column, candidate in enumerate(row["top_images"], 1):
                path = Path(candidate["image_path"])
                path = path if path.is_absolute() else root / path
                with Image.open(path) as image:
                    axes[row_index, column].imshow(image.convert("RGB"))
                marker = " ✓" if candidate["is_target"] else ""
                axes[row_index, column].set_title(f"Rank {candidate['rank']}{marker}\n{candidate['score']:.3f}")
                color = "limegreen" if candidate["is_target"] else "#555555"
                for spine in axes[row_index, column].spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(3 if candidate["is_target"] else 1)
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
            for column in range(len(row["top_images"]) + 1, topk + 1):
                axes[row_index, column].axis("off")
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()
