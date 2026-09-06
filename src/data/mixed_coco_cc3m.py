from __future__ import annotations

import csv
import hashlib
import io
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data.transforms import build_image_transform


def pass_permutation(length: int, *, seed: int, pass_index: int, arm_id: str) -> np.ndarray:
    digest = hashlib.sha256(
        f"{seed}:{pass_index}:{arm_id}:mixed-image-order-v1".encode()
    ).digest()
    generator = np.random.default_rng(
        int.from_bytes(digest[:8], "little", signed=False)
    )
    return generator.permutation(length).astype(np.int32)


class LogicalCursorSampler(Sampler[int]):
    def __init__(self, length: int, cursor: int = 0) -> None:
        if cursor < 0 or cursor > length:
            raise ValueError(f"cursor {cursor} outside 0..{length}")
        self.length, self.cursor = int(length), int(cursor)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.cursor, self.length))

    def __len__(self) -> int:
        return self.length - self.cursor


class MixedCocoCC3MDataset(Dataset[dict[str, object]]):
    """One image per logical sample; COCO retains all captions as positives."""

    def __init__(
        self,
        *,
        cc3m_index: str | Path,
        coco_csv: str | Path | None,
        arm_id: str,
        run_seed: int,
        pass_index: int,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        table = pq.read_table(
            cc3m_index,
            columns=[
                "canonical_id",
                "tar_path",
                "offset",
                "size",
                "caption",
                "raw_sha256",
            ],
            memory_map=True,
        ).to_pydict()
        self.cc3m = table
        self.cc3m_rows = len(table["canonical_id"])
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        if coco_csv is not None:
            with Path(coco_csv).open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not {
                    "image_path",
                    "caption",
                }.issubset(reader.fieldnames):
                    raise ValueError("COCO CSV must contain image_path and caption")
                for row in reader:
                    grouped.setdefault(str(row["image_path"]), []).append(
                        str(row["caption"])
                    )
        self.coco = list(grouped.items())
        self.arm_id = str(arm_id)
        self.run_seed = int(run_seed)
        self.pass_index = int(pass_index)
        self.length = self.cc3m_rows + len(self.coco)
        self.permutation = pass_permutation(
            self.length,
            seed=self.run_seed,
            pass_index=self.pass_index,
            arm_id=self.arm_id,
        )
        self.transform = build_image_transform(image_size=image_size, train=train)
        self._handles: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.length

    def _augment(self, image: Image.Image, key: str) -> torch.Tensor:
        digest = hashlib.sha256(
            f"{self.run_seed}:{self.pass_index}:{key}:augmentation-v2".encode()
        ).digest()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int.from_bytes(digest[:8], "little", signed=False))
            return self.transform(image)

    def _cc3m_item(self, index: int) -> dict[str, object]:
        path = str(self.cc3m["tar_path"][index])
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = open(path, "rb", buffering=0)
            self._handles[path] = handle
        offset, size = int(self.cc3m["offset"][index]), int(self.cc3m["size"][index])
        handle.seek(offset)
        payload = handle.read(size)
        if len(payload) != size:
            raise IOError(f"short tar read for {path}:{offset}")
        if hashlib.sha256(payload).hexdigest() != str(self.cc3m["raw_sha256"][index]):
            raise RuntimeError(f"CC3M raw hash mismatch at row {index}")
        key = str(self.cc3m["canonical_id"][index])
        with Image.open(io.BytesIO(payload)) as image:
            tensor = self._augment(image.convert("RGB"), key)
        return {
            "image": tensor,
            "caption": str(self.cc3m["caption"][index]),
            "image_path": key,
            "source": "cc3m",
        }

    def __getitem__(self, logical_index: int) -> dict[str, object]:
        source = int(self.permutation[int(logical_index)])
        if source < self.cc3m_rows:
            return self._cc3m_item(source)
        path_value, captions = self.coco[source - self.cc3m_rows]
        path = Path(path_value)
        with Image.open(path) as image:
            tensor = self._augment(image.convert("RGB"), str(path))
        return {
            "image": tensor,
            "caption": list(captions),
            "image_path": str(path),
            "source": "coco",
        }

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
