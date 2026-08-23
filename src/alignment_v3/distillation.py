from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.alignment_v3.fingerprint import Fingerprint, is_fresh, sha256_file, write_fingerprint
from src.phase15.io_utils import atomic_json


def _normalise_path(image_path: str) -> str:
    return str(Path(image_path).expanduser().resolve())


def _key(image_path: str, caption: str) -> str:
    return hashlib.sha256(f"{_normalise_path(image_path)}\0{caption}".encode("utf-8")).hexdigest()


class TeacherCache:
    def __init__(self, path: str | Path, *, verify_checksum: bool = True) -> None:
        self.path = Path(path)
        metadata_path = self.path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text())
        if verify_checksum and sha256_file(self.path) != self.metadata["content_sha256"]:
            raise ValueError("teacher cache content checksum mismatch")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        self.image_embeddings = payload["image_embeddings"]
        self.text_embeddings = payload["text_embeddings"]
        self.image_index = {
            _normalise_path(str(key)): index for index, key in enumerate(payload["image_paths"])
        }
        self.text_index = {str(key): index for index, key in enumerate(payload["text_keys"])}
        if self.image_embeddings.dtype != torch.float16 or self.text_embeddings.dtype != torch.float16:
            raise TypeError("teacher cache embeddings must be FP16")
        if len(self.image_index) != self.image_embeddings.shape[0]:
            raise ValueError("teacher image index is not one-to-one")
        if len(self.text_index) != self.text_embeddings.shape[0]:
            raise ValueError("teacher text index is not one-to-one")

    def lookup(
        self,
        image_paths: Sequence[str],
        text_image_paths: Sequence[str],
        captions: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(text_image_paths) != len(captions):
            raise ValueError("text image paths and captions must have equal lengths")
        try:
            image_indices = [self.image_index[_normalise_path(str(value))] for value in image_paths]
            text_indices = [
                self.text_index[_key(str(image_path), str(caption))]
                for image_path, caption in zip(text_image_paths, captions)
            ]
        except KeyError as exc:
            raise KeyError(f"teacher cache does not contain batch key {exc}") from exc
        images = self.image_embeddings[image_indices].to(device=device, non_blocking=True)
        texts = self.text_embeddings[text_indices].to(device=device, non_blocking=True)
        return images, texts


def save_teacher_cache(
    path: str | Path,
    *,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_paths: list[str],
    text_image_paths: list[str],
    captions: list[str],
    fingerprint: Fingerprint,
    teacher_id: str,
    logit_scale: float,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalised_image_paths = [_normalise_path(value) for value in image_paths]
    if len(set(normalised_image_paths)) != len(normalised_image_paths):
        raise ValueError("teacher image cache contains duplicate image paths")
    if len(text_image_paths) != len(captions) or len(captions) != text_embeddings.shape[0]:
        raise ValueError("teacher text embeddings, image paths and captions must align")
    raw_text_keys = [_key(path, caption) for path, caption in zip(text_image_paths, captions)]
    # COCO contains a small number of repeated captions for the same image.
    # Store one deterministic embedding per semantic key instead of allowing a
    # later duplicate to silently overwrite the lookup index.
    first_indices: list[int] = []
    seen: set[str] = set()
    for index, key in enumerate(raw_text_keys):
        if key not in seen:
            seen.add(key)
            first_indices.append(index)
    text_keys = [raw_text_keys[index] for index in first_indices]
    payload = {
        "image_embeddings": F.normalize(image_embeddings.float(), dim=-1).half().contiguous(),
        "text_embeddings": F.normalize(text_embeddings[first_indices].float(), dim=-1).half().contiguous(),
        "image_paths": normalised_image_paths,
        "text_keys": text_keys,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    metadata = {
        "status": "COMPLETE",
        "teacher_id": teacher_id,
        "image_rows": len(image_paths),
        "text_rows": len(text_keys),
        "embedding_dim": int(payload["image_embeddings"].shape[-1]),
        "dtype": str(payload["image_embeddings"].dtype),
        "logit_scale": float(logit_scale),
        "content_sha256": sha256_file(target),
        "fingerprint_digest": fingerprint.digest,
    }
    atomic_json(metadata, target.with_suffix(".metadata.json"))
    write_fingerprint(target.with_suffix(".fingerprint.json"), fingerprint)
    return metadata


def cache_is_fresh(path: str | Path, expected: Fingerprint) -> bool:
    target = Path(path)
    return (
        target.is_file()
        and target.with_suffix(".metadata.json").is_file()
        and is_fresh(target.with_suffix(".fingerprint.json"), expected)
    )


def symmetric_kl_distillation(
    student_image: torch.Tensor,
    student_text: torch.Tensor,
    teacher_image: torch.Tensor,
    teacher_text: torch.Tensor,
    *,
    temperature: float = 2.0,
    student_scale: torch.Tensor | float = 1.0,
    teacher_scale: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    si = F.normalize(student_image.float(), dim=-1)
    st = F.normalize(student_text.float(), dim=-1)
    ti = F.normalize(teacher_image.float(), dim=-1)
    tt = F.normalize(teacher_text.float(), dim=-1)
    s_scale = torch.as_tensor(student_scale, device=si.device, dtype=torch.float32).clamp(1e-3, 100)
    t_scale = float(max(1e-3, min(100.0, teacher_scale)))
    student_logits = (s_scale * si @ st.t() / temperature).clamp(-80, 80)
    teacher_logits = (t_scale * ti @ tt.t() / temperature).clamp(-80, 80)

    def direction(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        teacher_probability = F.softmax(teacher, dim=-1)
        student_log_probability = F.log_softmax(student, dim=-1)
        return F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")

    loss = 0.5 * (
        direction(student_logits, teacher_logits)
        + direction(student_logits.t(), teacher_logits.t())
    )
    loss = loss * temperature * temperature
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite symmetric KL distillation loss")
    return loss


def distillation_loss(
    outputs: dict[str, torch.Tensor],
    teacher_image: torch.Tensor,
    teacher_text: torch.Tensor,
    *,
    teacher_scale: float,
    temperature: float,
    image_cosine_weight: float,
    text_cosine_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_image = outputs["teacher_space_image"]
    student_text = outputs["teacher_space_text"]
    teacher_image = F.normalize(teacher_image.float(), dim=-1)
    teacher_text = F.normalize(teacher_text.float(), dim=-1)
    image_cosine = (1.0 - F.cosine_similarity(student_image.float(), teacher_image, dim=-1)).mean()
    text_cosine = (1.0 - F.cosine_similarity(student_text.float(), teacher_text, dim=-1)).mean()
    kl = symmetric_kl_distillation(
        student_image,
        student_text,
        teacher_image,
        teacher_text,
        temperature=temperature,
        student_scale=outputs["logit_scale"],
        teacher_scale=teacher_scale,
    )
    total = image_cosine_weight * image_cosine + text_cosine_weight * text_cosine + kl_weight * kl
    return total, {"image_cosine": image_cosine, "text_cosine": text_cosine, "kl": kl}
