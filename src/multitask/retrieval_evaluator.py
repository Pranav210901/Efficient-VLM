from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from src.training.metrics import compute_retrieval_metrics


def evaluate_retrieval(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, image_ids: Sequence[Hashable], text_image_ids: Sequence[Hashable]) -> dict[str, float]:
    metrics = compute_retrieval_metrics(image_embeddings, text_embeddings, (1, 5, 10), image_ids, text_image_ids)
    return {f"coco5_{key}": value for key, value in metrics.items()}


def retrieval_predictions(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, image_ids: Sequence[Hashable], text_image_ids: Sequence[Hashable], topk: int = 10) -> list[dict[str, object]]:
    unique_indices, unique_ids, seen = [], [], set()
    for index, value in enumerate(image_ids):
        if value not in seen:
            seen.add(value); unique_indices.append(index); unique_ids.append(value)
    images = image_embeddings[torch.tensor(unique_indices)]
    similarity = images @ text_embeddings.T
    rows = []
    for index, image_id in enumerate(unique_ids):
        order = similarity[index].argsort(descending=True)
        positives = torch.tensor([value == image_id for value in text_image_ids])
        rank = int(torch.nonzero(positives[order], as_tuple=False)[0, 0]) + 1
        pos_score = float(similarity[index, positives].max())
        neg_score = float(similarity[index, ~positives].max())
        chosen = order[: min(topk, len(order))]
        rows.append({"direction": "i2t", "query_id": str(image_id), "correct_target_rank": rank, "recall_at_1": rank <= 1, "recall_at_5": rank <= 5, "recall_at_10": rank <= 10, "positive_similarity": pos_score, "hardest_negative_similarity": neg_score, "retrieval_margin": pos_score - neg_score, "top_candidate_ids": [str(i) for i in chosen.tolist()], "top_candidate_scores": [float(similarity[index, i]) for i in chosen]})
    return rows


def text_to_image_predictions(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_ids: Sequence[Hashable],
    text_image_ids: Sequence[Hashable],
    topk: int = 5,
) -> list[dict[str, object]]:
    """Rank unique gallery images for each caption query."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("image_embeddings and text_embeddings must be two-dimensional")
    if image_embeddings.size(1) != text_embeddings.size(1):
        raise ValueError("image and text embedding dimensions must match")
    if len(image_ids) != image_embeddings.size(0):
        raise ValueError("image_ids must match image_embeddings")
    if len(text_image_ids) != text_embeddings.size(0):
        raise ValueError("text_image_ids must match text_embeddings")
    if topk < 1:
        raise ValueError("topk must be positive")

    unique_indices, unique_ids, seen = [], [], set()
    for index, value in enumerate(image_ids):
        if value not in seen:
            seen.add(value)
            unique_indices.append(index)
            unique_ids.append(value)
    gallery = image_embeddings.index_select(0, torch.tensor(unique_indices, device=image_embeddings.device))
    id_to_index = {value: index for index, value in enumerate(unique_ids)}
    try:
        targets = torch.tensor([id_to_index[value] for value in text_image_ids], device=gallery.device)
    except KeyError as exc:
        raise ValueError(f"caption references an image outside the gallery: {exc.args[0]}") from exc

    similarity = text_embeddings.to(gallery.device) @ gallery.t()
    rows = []
    for query_index in range(similarity.size(0)):
        order = similarity[query_index].argsort(descending=True)
        target = int(targets[query_index])
        rank = int(torch.nonzero(order.eq(target), as_tuple=False)[0, 0]) + 1
        chosen = order[: min(topk, len(order))]
        positive = float(similarity[query_index, target])
        negative_mask = torch.ones(similarity.size(1), dtype=torch.bool, device=similarity.device)
        negative_mask[target] = False
        negative_scores = similarity[query_index, negative_mask]
        negative_ids = torch.arange(similarity.size(1), device=similarity.device)[negative_mask]
        if negative_scores.numel():
            hardest_position = int(negative_scores.argmax())
            hardest_index = int(negative_ids[hardest_position])
            hardest = float(negative_scores[hardest_position])
            hardest_id: str | None = str(unique_ids[hardest_index])
        else:
            hardest = positive
            hardest_id = None
        rows.append(
            {
                "direction": "t2i",
                "query_id": f"caption:{query_index:06d}",
                "query_index": query_index,
                "target_image_id": str(text_image_ids[query_index]),
                "correct_target_rank": rank,
                "recall_at_1": rank <= 1,
                "recall_at_5": rank <= 5,
                "recall_at_10": rank <= 10,
                "positive_similarity": positive,
                "hardest_negative_similarity": hardest,
                "hardest_negative_id": hardest_id,
                "retrieval_margin": positive - hardest,
                "top_candidate_ids": [str(unique_ids[int(index)]) for index in chosen],
                "top_candidate_scores": [float(similarity[query_index, index]) for index in chosen],
            }
        )
    return rows
