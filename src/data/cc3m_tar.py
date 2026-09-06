from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sized

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data.transforms import build_image_transform


class CursorSampler(Sampler[int]):
    """Deterministic logical positions beginning at an exact resume cursor."""

    def __init__(self, length: int, cursor: int = 0) -> None:
        if cursor < 0 or cursor > length:
            raise ValueError(f"cursor {cursor} outside 0..{length}")
        self.length = int(length)
        self.cursor = int(cursor)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.cursor, self.length))

    def __len__(self) -> int:
        return self.length - self.cursor


def epoch_permutation(
    length: int,
    *,
    seed: int,
    pass_index: int,
    shards: list[str] | None = None,
) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{pass_index}:cc3m-order-v1".encode()).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    generator = np.random.default_rng(value)
    if shards is None:
        return generator.permutation(length).astype(np.int32)
    if len(shards) != length:
        raise ValueError("shard vector length differs from index length")
    by_shard: dict[str, list[int]] = {}
    for index, shard in enumerate(shards):
        by_shard.setdefault(str(shard), []).append(index)
    shard_order = np.asarray(sorted(by_shard), dtype=object)
    generator.shuffle(shard_order)
    values: list[np.ndarray] = []
    for shard in shard_order:
        positions = np.asarray(by_shard[str(shard)], dtype=np.int32)
        generator.shuffle(positions)
        values.append(positions)
    return np.concatenate(values)


class IndexedTarDataset(Dataset[dict[str, object]]):
    """Random-access WebDataset samples using tar member offsets, without extraction."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        run_seed: int,
        pass_index: int,
        image_size: int = 256,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        interpolation: str = "bicubic",
        train: bool = True,
        deterministic_order: bool = False,
    ) -> None:
        table = pq.read_table(
            index_path,
            columns=[
                "position",
                "canonical_id",
                "shard",
                "tar_path",
                "offset",
                "size",
                "caption",
                "raw_sha256",
            ],
            memory_map=True,
        )
        frame = table.to_pydict()
        positions = np.asarray(frame["position"], dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(positions), dtype=np.int64)):
            raise ValueError("CC3M tar index positions are not contiguous and ordered")
        self.canonical_ids = frame["canonical_id"]
        self.shards = frame["shard"]
        self.tar_paths = frame["tar_path"]
        self.offsets = np.asarray(frame["offset"], dtype=np.int64)
        self.sizes = np.asarray(frame["size"], dtype=np.int64)
        self.captions = frame["caption"]
        self.raw_sha256 = frame["raw_sha256"]
        self.run_seed = int(run_seed)
        self.pass_index = int(pass_index)
        self.permutation = (
            np.arange(len(positions), dtype=np.int32)
            if deterministic_order
            else epoch_permutation(
                len(positions),
                seed=self.run_seed,
                pass_index=self.pass_index,
                shards=self.shards,
            )
        )
        self.train = bool(train)
        self.transform = build_image_transform(
            image_size=image_size,
            train=self.train,
            mean=image_mean,
            std=image_std,
            interpolation=interpolation,
        )
        self._handles: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.permutation)

    def _read(self, tar_path: str, offset: int, size: int) -> bytes:
        handle = self._handles.get(tar_path)
        if handle is None or handle.closed:
            handle = open(tar_path, "rb", buffering=0)
            self._handles[tar_path] = handle
        handle.seek(offset)
        payload = handle.read(size)
        if len(payload) != size:
            raise IOError(f"short tar-member read from {tar_path}: {len(payload)} != {size}")
        return payload

    def __getitem__(self, logical_index: int) -> dict[str, object]:
        source = int(self.permutation[int(logical_index)])
        payload = self._read(
            str(self.tar_paths[source]),
            int(self.offsets[source]),
            int(self.sizes[source]),
        )
        if hashlib.sha256(payload).hexdigest() != str(self.raw_sha256[source]):
            raise RuntimeError(
                f"raw member hash mismatch: {self.canonical_ids[source]}"
            )
        with Image.open(io.BytesIO(payload)) as image:
            image.seek(0)
            rgb = image.convert("RGB")
        if self.train:
            augmentation_digest = hashlib.sha256(
                (
                    f"{self.run_seed}:{self.pass_index}:"
                    f"{self.canonical_ids[source]}:augmentation-v1"
                ).encode()
            ).digest()
            augmentation_seed = int.from_bytes(
                augmentation_digest[:8], "little", signed=False
            )
            # Each DataLoader worker is a separate process. Forking the CPU RNG
            # makes augmentation independent of prefetch depth and resume.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(augmentation_seed)
                tensor = self.transform(rgb)
        else:
            tensor = self.transform(rgb)
        return {
            "image": tensor,
            "caption": str(self.captions[source]),
            "image_path": str(self.canonical_ids[source]),
            "logical_position": int(logical_index),
        }

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def worker_init_noop(_: int) -> None:
    # Augmentation is sample-key seeded, so worker scheduling cannot change it.
    return None
