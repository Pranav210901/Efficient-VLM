from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from .config import PROMPT_TEMPLATES


def clean_class_name(class_name: str) -> str:
    # convert torchvision labels into natural text for prompt templates
    value = class_name.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return " ".join(value.split()).lower()


def class_prompts(class_name: str, templates: tuple[str, ...] = PROMPT_TEMPLATES) -> list[str]:
    cleaned = clean_class_name(class_name)
    return [template.format(class_name=cleaned) for template in templates]


@torch.no_grad()
def ensemble_class_embeddings(
    model,
    class_names: list[str],
    templates: tuple[str, ...] = PROMPT_TEMPLATES,
    batch_size: int = 256,
) -> torch.Tensor:
    # encode all prompts, then normalize-average-normalize each class
    if not class_names:
        raise ValueError("At least one class name is required")
    if not templates:
        raise ValueError("At least one prompt template is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    prompts = [prompt for name in class_names for prompt in class_prompts(name, templates)]
    parts = []
    for start in range(0, len(prompts), batch_size):
        encoded = model.encode_text(prompts[start : start + batch_size])
        if encoded.ndim != 2:
            raise ValueError("model.encode_text must return [prompts, embedding_dim]")
        parts.append(F.normalize(encoded, dim=-1))
    prompt_embeddings = torch.cat(parts, dim=0)
    grouped = prompt_embeddings.reshape(len(class_names), len(templates), -1)
    return F.normalize(grouped.mean(dim=1), dim=-1)
