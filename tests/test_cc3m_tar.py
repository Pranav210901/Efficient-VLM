from __future__ import annotations

import hashlib
import io
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.alignment_v3.training import _restore_rng, _rng_state
from src.data.cc3m_tar import CursorSampler, IndexedTarDataset, epoch_permutation
from src.data.collate import image_text_collate


def _fixture_index(tmp_path: Path) -> Path:
    tar_path = tmp_path / "fixture.tar"
    payloads = []
    for index, colour in enumerate(((10, 20, 30), (40, 50, 60), (70, 80, 90))):
        stream = io.BytesIO()
        Image.new("RGB", (320, 300), colour).save(stream, format="JPEG")
        payloads.append((f"k{index}", stream.getvalue()))
    with tarfile.open(tar_path, "w") as archive:
        for key, payload in payloads:
            info = tarfile.TarInfo(f"{key}.jpg")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    offsets = {}
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if member.name.endswith(".jpg"):
                offsets[Path(member.name).stem] = (member.offset_data, member.size)
    table = pa.table(
        {
            "position": list(range(3)),
            "canonical_id": [f"fixture.tar::k{i}" for i in range(3)],
            "shard": ["fixture.tar"] * 3,
            "tar_path": [str(tar_path)] * 3,
            "offset": [offsets[f"k{i}"][0] for i in range(3)],
            "size": [offsets[f"k{i}"][1] for i in range(3)],
            "caption": [f"caption number {i}" for i in range(3)],
            "raw_sha256": [
                hashlib.sha256(payloads[i][1]).hexdigest() for i in range(3)
            ],
        }
    )
    path = tmp_path / "index.parquet"
    pq.write_table(table, path)
    return path


def test_shard_local_epoch_order_is_deterministic() -> None:
    shards = ["a", "a", "b", "b", "c", "c"]
    left = epoch_permutation(6, seed=42, pass_index=0, shards=shards)
    right = epoch_permutation(6, seed=42, pass_index=0, shards=shards)
    assert left.tolist() == right.tolist()
    ordered_shards = [shards[index] for index in left]
    assert all(
        ordered_shards.count(shard)
        == max(index for index, value in enumerate(ordered_shards) if value == shard)
        - min(index for index, value in enumerate(ordered_shards) if value == shard)
        + 1
        for shard in set(shards)
    )


def test_tar_dataset_is_random_access_and_cursor_resumable(tmp_path: Path) -> None:
    index = _fixture_index(tmp_path)
    dataset = IndexedTarDataset(index, run_seed=42, pass_index=0, image_size=64)
    first = dataset[1]
    again = dataset[1]
    assert torch.equal(first["image"], again["image"])
    assert first["image_path"] == again["image_path"]
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=CursorSampler(len(dataset), cursor=1),
        num_workers=0,
        collate_fn=image_text_collate,
    )
    assert len(list(loader)) == 2


def test_deterministic_cache_transform_preserves_manifest_order(tmp_path: Path) -> None:
    index = _fixture_index(tmp_path)
    dataset = IndexedTarDataset(
        index,
        run_seed=0,
        pass_index=0,
        image_size=64,
        train=False,
        deterministic_order=True,
    )
    assert [dataset[index]["image_path"] for index in range(3)] == [
        f"fixture.tar::k{index}" for index in range(3)
    ]
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])


def _first_divergence(left: Any, right: Any, path: str = "root") -> str | None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            return (
                f"{path}: tensor metadata {left.dtype}/{tuple(left.shape)} != "
                f"{right.dtype}/{tuple(right.shape)}"
            )
        if not torch.equal(left, right):
            differing = torch.nonzero(left != right)
            first = tuple(int(value) for value in differing[0].tolist())
            return f"{path}{first}: {left[first].item()} != {right[first].item()}"
        return None
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if list(left) != list(right):
            return f"{path}: keys {list(left)} != {list(right)}"
        for key in left:
            found = _first_divergence(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            found = _first_divergence(
                left_value, right_value, f"{path}[{index}]"
            )
            if found:
                return found
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def test_mid_epoch_resume_is_bitwise_exact_at_step_737_of_1320() -> None:
    """Strict recovery contract for the state that protects CC3M Arm A."""
    total_steps = 1320
    interruption_step = 737
    keys = [f"sample-{index:04d}" for index in range(total_steps)]
    order_generator = torch.Generator().manual_seed(20260729)
    order = torch.randperm(total_steps, generator=order_generator).tolist()

    def initialise():
        torch.manual_seed(123)
        projector = torch.nn.Sequential(
            torch.nn.Linear(6, 8),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(8, 2),
        )
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
        return projector, optimizer, scheduler

    def step(projector, optimizer, scheduler, source_index):
        base = torch.arange(6, dtype=torch.float32) + source_index / 1000
        augmented = base + torch.rand_like(base) * 0.01
        target = torch.tensor(
            [source_index % 7, source_index % 11], dtype=torch.float32
        ) / 10
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(projector(augmented), target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    def run_until(projector, optimizer, scheduler, start, stop, consumed):
        for logical_step in range(start, stop):
            source_index = order[logical_step]
            consumed.append(keys[source_index])
            step(projector, optimizer, scheduler, source_index)

    full_projector, full_optimizer, full_scheduler = initialise()
    full_keys: list[str] = []
    run_until(
        full_projector, full_optimizer, full_scheduler, 0, total_steps, full_keys
    )

    interrupted_projector, interrupted_optimizer, interrupted_scheduler = initialise()
    resumed_keys: list[str] = []
    run_until(
        interrupted_projector,
        interrupted_optimizer,
        interrupted_scheduler,
        0,
        interruption_step,
        resumed_keys,
    )
    checkpoint = deepcopy(
        {
            "projector": interrupted_projector.state_dict(),
            "optimizer": interrupted_optimizer.state_dict(),
            "scheduler": interrupted_scheduler.state_dict(),
            "rng": _rng_state(),
            "cursor": interruption_step,
            "consumed_keys": resumed_keys,
        }
    )
    resumed_projector, resumed_optimizer, resumed_scheduler = initialise()
    resumed_projector.load_state_dict(checkpoint["projector"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_rng(checkpoint["rng"])
    resumed_keys = list(checkpoint["consumed_keys"])
    run_until(
        resumed_projector,
        resumed_optimizer,
        resumed_scheduler,
        checkpoint["cursor"],
        total_steps,
        resumed_keys,
    )

    comparisons = {
        "projector_weights": (
            full_projector.state_dict(),
            resumed_projector.state_dict(),
        ),
        "optimizer_state": (
            full_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        ),
        "scheduler_state": (
            full_scheduler.state_dict(),
            resumed_scheduler.state_dict(),
        ),
        "consumed_sample_keys": (full_keys, resumed_keys),
    }
    for name, (left, right) in comparisons.items():
        divergence = _first_divergence(left, right, name)
        assert divergence is None, (
            f"resume-integrity failure; first divergence: {divergence}"
        )
