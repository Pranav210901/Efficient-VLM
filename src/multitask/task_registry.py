from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    group: str
    mandatory: bool = True


TASK_REGISTRY = {
    "coco_retrieval": TaskSpec("coco_retrieval", "retrieval"),
    "cifar100_zeroshot": TaskSpec("cifar100_zeroshot", "classification"),
    "pets_zeroshot": TaskSpec("pets_zeroshot", "classification"),
    "eurosat_zeroshot": TaskSpec("eurosat_zeroshot", "classification"),
    "imagenet1k_zeroshot": TaskSpec("imagenet1k_zeroshot", "classification", False),
    "winoground": TaskSpec("winoground", "compositional", False),
    "sugarcrepe": TaskSpec("sugarcrepe", "compositional", False),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown multi-task dataset {name!r}; choose from {sorted(TASK_REGISTRY)}") from exc
