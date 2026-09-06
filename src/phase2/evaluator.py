from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_image_transform
from src.phase2.schemas import TokenBatch
from src.multitask.classification_evaluator import prepare_zeroshot_dataset


class ImagePaths(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.transform = build_image_transform(image_size, train=False)
    def __len__(self): return len(self.paths)
    def __getitem__(self, index): return self.transform(Image.open(self.paths[index]).convert("RGB"))


def read_coco_captions(csv_path: str | Path, root: str | Path) -> tuple[list[str], list[str], list[str]]:
    base = Path(root)
    grouped: dict[str, list[str]] = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = Path(row["image_path"])
            path = str(raw if raw.is_absolute() else base / raw)
            grouped.setdefault(path, []).append(str(row["caption"]))
    images = list(grouped)
    captions = [caption for path in images for caption in grouped[path]]
    owners = [path for path in images for _ in grouped[path]]
    return images, captions, owners


def _pad_concat(batches: list[TokenBatch]) -> TokenBatch:
    maximum = max(batch.tokens.shape[1] for batch in batches)
    values, masks = [], []
    for batch in batches:
        padding = maximum - batch.tokens.shape[1]
        values.append(torch.nn.functional.pad(batch.tokens, (0, 0, 0, padding)))
        masks.append(torch.nn.functional.pad(batch.attention_mask, (0, padding), value=False))
    tokens, mask = torch.cat(values), torch.cat(masks)
    return TokenBatch(tokens, mask, None, {}).validate()


@torch.no_grad()
def cache_projected_tokens(model, image_paths: list[str], captions: list[str], image_size: int, batch_size: int, device: torch.device) -> tuple[TokenBatch, TokenBatch]:
    vision_parts = []
    loader = DataLoader(ImagePaths(image_paths, image_size), batch_size=batch_size, num_workers=4, pin_memory=device.type == "cuda")
    for images in loader:
        batch = model.tokens.vision_tokens(images.to(device))
        vision_parts.append(TokenBatch(batch.tokens.cpu(), batch.attention_mask.cpu(), None, batch.metadata))
    text_parts = []
    for start in range(0, len(captions), batch_size):
        batch = model.tokens.text_tokens(captions[start:start + batch_size])
        text_parts.append(TokenBatch(batch.tokens.cpu(), batch.attention_mask.cpu(), None, batch.metadata))
    return _pad_concat(vision_parts), _pad_concat(text_parts)


def _select(batch: TokenBatch, indices: torch.Tensor, device: torch.device) -> TokenBatch:
    indices = indices.cpu()
    return TokenBatch(batch.tokens.index_select(0, indices).to(device), batch.attention_mask.index_select(0, indices).to(device), None, batch.metadata)


@torch.no_grad()
def _cross_scores(model, vision: TokenBatch, text: TokenBatch, image_indices: torch.Tensor, text_indices: torch.Tensor, device: torch.device, pair_batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(image_indices), pair_batch_size):
        end = start + pair_batch_size
        output = model.forward_tokens(_select(vision, image_indices[start:end], device), _select(text, text_indices[start:end], device))
        parts.append(output["scores"].float().cpu())
    return torch.cat(parts)


def _dual_topk(query: torch.Tensor, gallery: torch.Tensor, depth: int, batch_size: int = 512) -> torch.Tensor:
    parts = []
    gallery = gallery.float()
    for start in range(0, len(query), batch_size):
        parts.append((query[start:start + batch_size].float() @ gallery.T).topk(min(depth, len(gallery)), dim=1).indices)
    return torch.cat(parts)


def _standardise(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


def _fused_scores(dual: torch.Tensor, cross: torch.Tensor, dual_score_weight: float) -> torch.Tensor:
    weight = float(dual_score_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("dual_score_weight must be between zero and one")
    if weight == 0.0:
        return cross
    if weight == 1.0:
        return dual
    return weight * _standardise(dual) + (1.0 - weight) * _standardise(cross)


def evaluate_coco_reranking(model, cache_payload: dict, csv_path: str | Path, root: str | Path, depths: Iterable[int], image_size: int, batch_size: int, pair_batch_size: int, device: str | torch.device, dual_score_weight: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = torch.device(device)
    image_paths, captions, owners = read_coco_captions(csv_path, root)
    image_embeddings = cache_payload["image_embeddings"].cpu()
    text_embeddings = cache_payload["text_embeddings"].cpu()
    if list(map(str, cache_payload["image_ids"])) != image_paths or list(map(str, cache_payload["text_image_ids"])) != owners:
        raise ValueError("COCO cache IDs do not align with the configured validation CSV")
    maximum = max(map(int, depths))
    i2t_candidates = _dual_topk(image_embeddings, text_embeddings, maximum)
    t2i_candidates = _dual_topk(text_embeddings, image_embeddings, maximum)
    i2t_dual = (image_embeddings.float() @ text_embeddings.float().t()).gather(1, i2t_candidates)
    t2i_dual = (text_embeddings.float() @ image_embeddings.float().t()).gather(1, t2i_candidates)
    vision, text = cache_projected_tokens(model, image_paths, captions, image_size, batch_size, target)
    results, samples, coverage = [], [], []
    image_lookup = {path: index for index, path in enumerate(image_paths)}
    text_owner_indices = torch.tensor([image_lookup[value] for value in owners])
    for direction, candidates, dual in (("i2t", i2t_candidates, i2t_dual), ("t2i", t2i_candidates, t2i_dual)):
        query_count = candidates.shape[0]
        query_indices = torch.arange(query_count).unsqueeze(1).expand_as(candidates)
        image_indices, text_indices = (query_indices, candidates) if direction == "i2t" else (candidates, query_indices)
        started = time.perf_counter()
        cross = _cross_scores(model, vision, text, image_indices.reshape(-1), text_indices.reshape(-1), target, pair_batch_size).reshape_as(candidates)
        elapsed = time.perf_counter() - started
        for depth in sorted(set(map(int, depths))):
            subset_candidates = candidates[:, :depth]
            subset_scores = _fused_scores(dual[:, :depth], cross[:, :depth], dual_score_weight)
            ranked = subset_candidates.gather(1, subset_scores.argsort(dim=1, descending=True))
            if direction == "i2t":
                valid = text_owner_indices[ranked].eq(torch.arange(query_count).unsqueeze(1))
            else:
                valid = ranked.eq(text_owner_indices.unsqueeze(1))
            present = valid.any(1)
            rank = torch.where(present, valid.float().argmax(1) + 1, torch.full((query_count,), depth + 1))
            metrics = {f"recall_at_{k}": float(rank.le(k).float().mean()) for k in (1, 5, 10)}
            metrics.update({"candidate_coverage": float(present.float().mean()), "mean_rank": float(rank.float().mean()), "median_rank": float(rank.float().median()), "reranking_latency_ms": elapsed * 1000 / query_count, "pair_count": query_count * depth})
            results.extend({"direction": direction, "candidate_depth": depth, "metric": key, "value": value} for key, value in metrics.items())
            coverage.append({"direction": direction, "candidate_depth": depth, "candidate_coverage": metrics["candidate_coverage"], "queries": query_count})
            samples.extend({"direction": direction, "candidate_depth": depth, "query_index": i, "positive_rank": int(rank[i]), "covered": bool(present[i])} for i in range(query_count))
    return pd.DataFrame(results), pd.DataFrame(samples), pd.DataFrame(coverage)


@torch.no_grad()
def evaluate_classification_reranking(model, cache_payload: dict, task: str, root: str | Path, depths: Iterable[int], image_size: int, batch_size: int, pair_batch_size: int, device: str | torch.device, dual_score_weight: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_device = torch.device(device)
    dataset, class_names = prepare_zeroshot_dataset(root, task, image_size, split_role="development")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=target_device.type == "cuda")
    vision_parts, targets, sample_ids = [], [], []
    for images, labels, identifiers in loader:
        batch = model.tokens.vision_tokens(images.to(target_device))
        vision_parts.append(TokenBatch(batch.tokens.cpu(), batch.attention_mask.cpu(), None, batch.metadata))
        targets.append(labels.cpu()); sample_ids.extend(map(str, identifiers))
    vision = _pad_concat(vision_parts); labels = torch.cat(targets)
    prompts = [f"a photo of a {name}" for name in class_names]
    text_parts = []
    for start in range(0, len(prompts), batch_size):
        batch = model.tokens.text_tokens(prompts[start:start + batch_size])
        text_parts.append(TokenBatch(batch.tokens.cpu(), batch.attention_mask.cpu(), None, batch.metadata))
    text = _pad_concat(text_parts)
    logits = cache_payload["logits"].float().cpu()
    if sample_ids != list(map(str, cache_payload["sample_ids"])) or not torch.equal(labels, cache_payload["targets"].cpu()):
        raise ValueError(f"{task} development cache does not align with the registered development dataset")
    maximum = min(max(map(int, depths)), logits.shape[1]); candidates = logits.topk(maximum, dim=1).indices
    sample_indices = torch.arange(len(dataset)).unsqueeze(1).expand_as(candidates)
    started = time.perf_counter()
    cross = _cross_scores(model, vision, text, sample_indices.reshape(-1), candidates.reshape(-1), target_device, pair_batch_size).reshape_as(candidates)
    elapsed = time.perf_counter() - started
    results, samples, coverage = [], [], []
    for requested in sorted(set(map(int, depths))):
        depth = min(requested, logits.shape[1]); subset = candidates[:, :depth]
        dual = logits.gather(1, subset)
        scores = _fused_scores(dual, cross[:, :depth], dual_score_weight)
        ranked = subset.gather(1, scores.argsort(dim=1, descending=True)); matches = ranked.eq(labels.unsqueeze(1)); present = matches.any(1)
        metrics = {
            "candidate_coverage": float(present.float().mean()), "top1": float(ranked[:, :1].eq(labels.unsqueeze(1)).any(1).float().mean()),
            "top5": float(ranked[:, : min(5, depth)].eq(labels.unsqueeze(1)).any(1).float().mean()),
            "additional_latency_ms": elapsed * 1000 / len(dataset), "candidate_count": float(depth),
        }
        results.extend({"task": task, "candidate_depth": requested, "metric": key, "value": value} for key, value in metrics.items())
        coverage.append({"task": task, "candidate_depth": requested, "candidate_coverage": metrics["candidate_coverage"], "samples": len(dataset)})
        samples.extend({"task": task, "candidate_depth": requested, "query_index": index, "sample_id": sample_ids[index], "target": int(labels[index]), "prediction": int(ranked[index, 0]), "correct": bool(ranked[index, 0] == labels[index]), "covered": bool(present[index])} for index in range(len(dataset)))
    return pd.DataFrame(results), pd.DataFrame(samples), pd.DataFrame(coverage)
