from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch


def _directional_metrics(similarity: torch.Tensor, k_values: Sequence[int], prefix: str) -> dict[str, float]:
    n = similarity.size(0)
    target = torch.arange(n, device=similarity.device)
    order = similarity.argsort(dim=1, descending=True)
    hits = order.eq(target.unsqueeze(1))
    ranks = hits.float().argmax(dim=1) + 1
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"{prefix}_R@{k}"] = float((ranks <= k).float().mean().item())
    metrics[f"mean_rank_{prefix}"] = float(ranks.float().mean().item())
    metrics[f"median_rank_{prefix}"] = float(ranks.float().median().item())
    return metrics


def _positive_mask_metrics(
    similarity: torch.Tensor,
    positive_mask: torch.Tensor,
    k_values: Sequence[int],
    prefix: str,
) -> dict[str, float]:
    if similarity.shape != positive_mask.shape:
        raise ValueError("similarity and positive_mask must have the same shape")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every query must have at least one positive target")
    order = similarity.argsort(dim=1, descending=True)
    ordered_positives = positive_mask.gather(1, order)
    ranks = ordered_positives.float().argmax(dim=1) + 1
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"{prefix}_R@{k}"] = float((ranks <= k).float().mean().item())
    metrics[f"mean_rank_{prefix}"] = float(ranks.float().mean().item())
    metrics[f"median_rank_{prefix}"] = float(ranks.float().median().item())
    return metrics


def _deduplicate_images(
    image_embeds: torch.Tensor,
    image_ids: Sequence[Hashable],
) -> tuple[torch.Tensor, list[Hashable]]:
    if len(image_ids) != image_embeds.size(0):
        raise ValueError("image_ids must match the number of image embeddings")
    first_indices: list[int] = []
    unique_ids: list[Hashable] = []
    seen: set[Hashable] = set()
    for index, image_id in enumerate(image_ids):
        if image_id in seen:
            continue
        seen.add(image_id)
        first_indices.append(index)
        unique_ids.append(image_id)
    index_tensor = torch.tensor(first_indices, device=image_embeds.device)
    return image_embeds.index_select(0, index_tensor), unique_ids


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: Sequence[int] = (1, 5, 10),
    image_ids: Sequence[Hashable] | None = None,
    text_image_ids: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    if (image_ids is None) != (text_image_ids is None):
        raise ValueError("image_ids and text_image_ids must be provided together")
    if image_ids is not None and text_image_ids is not None:
        if len(text_image_ids) != text_embeds.size(0):
            raise ValueError("text_image_ids must match the number of text embeddings")
        unique_image_embeds, unique_image_ids = _deduplicate_images(image_embeds, image_ids)
        similarity = unique_image_embeds @ text_embeds.t()
        image_id_to_index = {image_id: index for index, image_id in enumerate(unique_image_ids)}
        try:
            text_targets = torch.tensor(
                [image_id_to_index[text_image_id] for text_image_id in text_image_ids],
                device=similarity.device,
            )
        except KeyError as exc:
            raise ValueError(f"text caption references an unknown image ID: {exc.args[0]}") from exc
        query_indices = torch.arange(len(unique_image_ids), device=similarity.device)
        positive_mask = query_indices.unsqueeze(1).eq(text_targets.unsqueeze(0))
        metrics = _positive_mask_metrics(similarity, positive_mask, k_values, "i2t")
        metrics.update(_positive_mask_metrics(similarity.t(), positive_mask.t(), k_values, "t2i"))
        metrics["num_image_queries"] = float(len(unique_image_ids))
        metrics["num_text_queries"] = float(len(text_image_ids))
        return metrics

    if image_embeds.size(0) != text_embeds.size(0):
        raise ValueError("one-to-one retrieval requires equal image and text counts")
    similarity = image_embeds @ text_embeds.t()
    metrics = _directional_metrics(similarity, k_values, "i2t")
    metrics.update(_directional_metrics(similarity.t(), k_values, "t2i"))
    return metrics
