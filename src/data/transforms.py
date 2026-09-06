from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalisation(values: Sequence[float] | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    selected = default if values is None else tuple(float(value) for value in values)
    if len(selected) != 3:
        raise ValueError("image normalisation must contain three channel values")
    return selected


def build_image_transform(
    image_size: int = 224,
    train: bool = True,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    interpolation: str = "bicubic",
) -> transforms.Compose:
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    if interpolation not in interpolation_modes:
        raise ValueError(f"unsupported interpolation {interpolation!r}")
    mode = interpolation_modes[interpolation]
    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), interpolation=mode),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.Resize(image_size, interpolation=mode),
            transforms.CenterCrop(image_size),
        ]
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                _normalisation(mean, IMAGENET_MEAN),
                _normalisation(std, IMAGENET_STD),
            ),
        ]
    )
    return transforms.Compose(ops)
