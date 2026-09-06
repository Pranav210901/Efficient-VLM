from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import socket
import statistics
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError

from src.alignment_v3.fingerprint import sha256_file
from src.phase15.io_utils import atomic_json
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = "configs/cc3m_scale/pipeline.yaml"


def _pipeline(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return load_config(value if value.is_absolute() else ROOT / value)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _root(pipeline: dict[str, Any]) -> Path:
    return _path(pipeline["root"])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _normalise_caption(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _contains_meaningful_text(value: str) -> bool:
    return any(not unicodedata.category(char).startswith("C") for char in value)


def _pixel_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(
        ordered[low] * (high - position) + ordered[high] * (position - low)
    )


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)) if values else float("nan"),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def mirror_preflight(pipeline: dict[str, Any]) -> dict[str, Any]:
    spec = pipeline["acquisition"]
    api = HfApi()
    info = api.dataset_info(
        str(spec["repo_id"]),
        revision=str(spec["revision"]),
        files_metadata=True,
    )
    if info.sha != str(spec["revision"]):
        raise RuntimeError(f"mirror revision resolved to {info.sha}, not pinned revision")
    files = {}
    for sibling in info.siblings or []:
        name = str(sibling.rfilename)
        if name.startswith("cc3m-train-") and name.endswith(".tar"):
            lfs = sibling.lfs
            files[name] = {
                "bytes": int(sibling.size),
                "upstream_sha256": str(lfs.sha256) if lfs is not None else None,
            }
    expected = int(spec["mirror_train_shards"])
    if len(files) != expected or any(not row["upstream_sha256"] for row in files.values()):
        raise RuntimeError(
            f"pinned mirror exposes {len(files)} hashable train shards; expected {expected}"
        )
    ordered = sorted(files)
    random.Random(int(spec["shuffle_seed"])).shuffle(ordered)
    selected = ordered[: int(spec["download_train_shards"])]
    manifest = {
        "status": "READY",
        "repo_id": spec["repo_id"],
        "revision": info.sha,
        "format": "WebDataset tar shards",
        "mirror_train_samples": int(spec["mirror_train_samples"]),
        "original_cc3m_urls": int(spec["original_cc3m_urls"]),
        "mirror_train_shards": len(files),
        "download_train_shards": len(selected),
        "shuffle_seed": int(spec["shuffle_seed"]),
        "selected_shards_in_seed_order": selected,
        "selected_total_bytes": sum(files[name]["bytes"] for name in selected),
        "selected_shards": [
            {"seed_order": index, "filename": name, **files[name]}
            for index, name in enumerate(selected)
        ],
        "not_downloaded_shards": [name for name in ordered if name not in set(selected)],
        "provenance_limitation": (
            "This is CC3M as distributed by pixparse/cc3m-wds at the pinned "
            "revision, not the original CC3M distribution. It inherits 2021 URL "
            "survival, img2dataset filtering, and mirror-time resizing."
        ),
    }
    atomic_json(manifest, _root(pipeline) / "manifests/mirror_preflight.json")
    return manifest


def _preflight(pipeline: dict[str, Any]) -> dict[str, Any]:
    path = _root(pipeline) / "manifests/mirror_preflight.json"
    if not path.is_file():
        return mirror_preflight(pipeline)
    payload = json.loads(path.read_text())
    if (
        payload.get("revision") != pipeline["acquisition"]["revision"]
        or payload.get("download_train_shards")
        != int(pipeline["acquisition"]["download_train_shards"])
    ):
        raise RuntimeError("stale mirror preflight manifest")
    return payload


def download_shard(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    manifest = _preflight(pipeline)
    selected = (
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if index is None
        else int(index)
    )
    shards = manifest["selected_shards"]
    if selected < 0 or selected >= len(shards):
        raise IndexError(f"shard index {selected} outside 0..{len(shards) - 1}")
    shard = shards[selected]
    destination = _root(pipeline) / "shards/train" / shard["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == int(shard["bytes"])
        and sha256_file(destination) == shard["upstream_sha256"]
    ):
        action = "REUSED_VERIFIED"
    else:
        destination.unlink(missing_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=str(pipeline["acquisition"]["repo_id"]),
                repo_type="dataset",
                revision=str(pipeline["acquisition"]["revision"]),
                filename=str(shard["filename"]),
                local_dir=str(destination.parent),
            )
        )
        if downloaded.resolve() != destination.resolve():
            os.replace(downloaded, destination)
        digest = sha256_file(destination)
        if digest != shard["upstream_sha256"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {shard['filename']}")
        action = "DOWNLOADED_VERIFIED"
    result = {
        "status": "COMPLETE",
        "seed_order": selected,
        "filename": shard["filename"],
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "upstream_sha256": shard["upstream_sha256"],
        "revision": pipeline["acquisition"]["revision"],
        "action": action,
    }
    atomic_json(
        result,
        _root(pipeline) / f"manifests/download_{selected:03d}.json",
    )
    return result


def _sample_members(path: Path) -> Iterable[tuple[str, dict[str, bytes]]]:
    current_key: str | None = None
    fields: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if "." not in name:
                continue
            key, extension = name.rsplit(".", 1)
            extension = extension.lower()
            if current_key is not None and key != current_key:
                yield current_key, fields
                fields = {}
            current_key = key
            extracted = archive.extractfile(member)
            if extracted is not None:
                fields[extension] = extracted.read()
        if current_key is not None:
            yield current_key, fields


def _classify_sample(
    pipeline: dict[str, Any], shard: dict[str, Any], key: str, fields: dict[str, bytes]
) -> dict[str, Any]:
    spec = pipeline["acquisition"]
    base = {
        "seed_order": int(shard["seed_order"]),
        "shard": shard["filename"],
        "key": key,
        "canonical_id": f"{shard['filename']}::{key}",
    }
    image_fields = [name for name in ("jpg", "jpeg", "png", "webp") if name in fields]
    if len(image_fields) != 1 or "txt" not in fields:
        return {**base, "status": "REJECTED", "rejection": "missing_or_ambiguous_fields"}
    raw = fields[image_fields[0]]
    try:
        caption = _normalise_caption(fields["txt"].decode("utf-8"))
    except UnicodeDecodeError:
        return {**base, "status": "REJECTED", "rejection": "caption_utf8"}
    tokens = len(caption.split())
    if not caption or not _contains_meaningful_text(caption):
        return {**base, "status": "REJECTED", "rejection": "caption_empty_or_control"}
    if len(caption) > int(spec["caption_maximum_characters"]):
        return {**base, "status": "REJECTED", "rejection": "caption_characters"}
    if not (
        int(spec["caption_minimum_whitespace_tokens"])
        <= tokens
        <= int(spec["caption_maximum_whitespace_tokens"])
    ):
        return {**base, "status": "REJECTED", "rejection": "caption_tokens"}
    try:
        image = Image.open(io.BytesIO(raw))
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, EOFError) as exc:
        return {
            **base,
            "status": "REJECTED",
            "rejection": "decode_or_frame_zero",
            "error": type(exc).__name__,
        }
    width, height = image.size
    common = {
        **base,
        "caption": caption,
        "caption_tokens": tokens,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "pixel_sha256": _pixel_digest(image),
        "width": width,
        "height": height,
        "short_side": min(width, height),
        "aspect_ratio": width / height,
    }
    if width * height > int(spec["maximum_pixels"]):
        rejection = "maximum_pixels"
    elif min(width, height) < int(spec["minimum_short_side"]):
        rejection = "minimum_resolution"
    elif not (
        float(spec["minimum_aspect_ratio"])
        <= width / height
        <= float(spec["maximum_aspect_ratio"])
    ):
        rejection = "aspect_ratio"
    else:
        rejection = ""
    if rejection:
        return {**common, "status": "REJECTED", "rejection": rejection}
    return {**common, "status": "USABLE", "rejection": None}


def inspect_shard(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    manifest = _preflight(pipeline)
    selected = (
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if index is None
        else int(index)
    )
    shard = manifest["selected_shards"][selected]
    tar_path = _root(pipeline) / "shards/train" / shard["filename"]
    download_manifest = _root(pipeline) / f"manifests/download_{selected:03d}.json"
    ledger = _root(pipeline) / f"ledgers/inspect_{selected:03d}.jsonl"
    summary_path = _root(pipeline) / f"manifests/inspect_{selected:03d}.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        if (
            existing.get("tar_sha256") == shard["upstream_sha256"]
            and existing.get("ledger_sha256") == sha256_file(ledger)
        ):
            return {**existing, "action": "REUSED_VERIFIED"}
        raise RuntimeError(f"stale inspection artifact for {shard['filename']}")
    if not tar_path.is_file() or not download_manifest.is_file():
        raise FileNotFoundError(f"download not complete for {shard['filename']}")
    existing_rows: dict[str, dict[str, Any]] = {}
    if ledger.is_file():
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = str(row["key"])
                if key in existing_rows:
                    raise RuntimeError(
                        f"append-only ledger contains duplicate key {key}: {ledger}"
                    )
                existing_rows[key] = row
    resumed_records = len(existing_rows)
    keys: set[str] = set()
    for key, fields in _sample_members(tar_path):
        if key in keys:
            raise RuntimeError(f"duplicate bare key within {shard['filename']}: {key}")
        keys.add(key)
        if key in existing_rows:
            continue
        row = _classify_sample(pipeline, shard, key, fields)
        _append_jsonl(ledger, row)
        existing_rows[key] = row
    if keys != set(existing_rows):
        raise RuntimeError(
            f"ledger/tar key mismatch for {shard['filename']}: "
            f"tar={len(keys)} ledger={len(existing_rows)}"
        )
    counts = Counter()
    caption_tokens: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    short_sides: list[float] = []
    for row in existing_rows.values():
        counts[row["status"]] += 1
        if row["status"] == "REJECTED":
            counts[f"rejected:{row['rejection']}"] += 1
        if "caption_tokens" in row:
            caption_tokens.append(float(row["caption_tokens"]))
        if "width" in row:
            widths.append(float(row["width"]))
            heights.append(float(row["height"]))
            short_sides.append(float(row["short_side"]))
    total = counts["USABLE"] + counts["REJECTED"]
    result = {
        "status": "COMPLETE",
        "seed_order": selected,
        "filename": shard["filename"],
        "tar_sha256": sha256_file(tar_path),
        "ledger": str(ledger.relative_to(ROOT)),
        "ledger_sha256": sha256_file(ledger),
        "samples_examined": total,
        "usable": counts["USABLE"],
        "rejected": counts["REJECTED"],
        "rejection_rate": counts["REJECTED"] / max(1, total),
        "rejection_counts": {
            key.removeprefix("rejected:"): value
            for key, value in counts.items()
            if key.startswith("rejected:")
        },
        "caption_tokens": _summary(caption_tokens),
        "width": _summary(widths),
        "height": _summary(heights),
        "short_side": _summary(short_sides),
        "bare_keys_unique_within_shard": True,
        "resumed_records": resumed_records,
        "minimum_resolution_rejections": counts["rejected:minimum_resolution"],
    }
    atomic_json(result, summary_path)
    return result


def _csv_image_paths(paths: Iterable[str]) -> set[Path]:
    images: set[Path] = set()
    for value in paths:
        path = _path(value)
        if not path.is_file():
            raise FileNotFoundError(f"deduplication source missing: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                images.add(_path(row["image_path"]))
    return images


def _exclusion_hashes(pipeline: dict[str, Any]) -> tuple[set[str], set[str], dict[str, Any]]:
    paths = _csv_image_paths(pipeline["acquisition"]["exclusion_csvs"])
    raw_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    failures = 0
    for path in sorted(paths):
        if not path.is_file():
            failures += 1
            continue
        try:
            raw_hashes.add(sha256_file(path))
            with Image.open(path) as image:
                image.seek(0)
                pixel_hashes.add(_pixel_digest(ImageOps.exif_transpose(image)))
        except (OSError, ValueError, EOFError):
            failures += 1
    return raw_hashes, pixel_hashes, {
        "source_images": len(paths),
        "raw_hashes": len(raw_hashes),
        "pixel_hashes": len(pixel_hashes),
        "missing_or_invalid": failures,
        "csvs": list(pipeline["acquisition"]["exclusion_csvs"]),
    }


def _heterogeneity(
    pipeline: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    count = int(pipeline["acquisition"]["heterogeneity_shards"])
    audited = summaries[:count]
    statistics_rows: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []
    for metric in ("caption_tokens", "width", "height", "short_side"):
        for statistic_name in ("mean", "p50", "p95"):
            values = [float(row[metric][statistic_name]) for row in audited]
            pooled_mean = statistics.fmean(values)
            range_value = max(values) - min(values)
            cv = statistics.pstdev(values) / pooled_mean if pooled_mean else 0.0
            # The frozen rule applies to the range of per-shard means. The
            # p50/p95 ranges and CVs are still always reported descriptively.
            flagged = (
                statistic_name == "mean" and range_value > 0.10 * pooled_mean
            )
            key = f"{metric}_{statistic_name}"
            statistics_rows[key] = {
                "pooled_mean_of_shard_statistics": pooled_mean,
                "range_across_shards": range_value,
                "range_fraction_of_pooled_mean": (
                    range_value / pooled_mean if pooled_mean else None
                ),
                "coefficient_of_variation": cv,
                "flagged": flagged,
            }
            if flagged:
                flags.append({"kind": "distribution_range", "statistic": key})
    pooled_rejection = sum(row["rejected"] for row in audited) / max(
        1, sum(row["samples_examined"] for row in audited)
    )
    categories = sorted(
        {
            category
            for row in audited
            for category in row["rejection_counts"]
        }
    )
    category_rows = {}
    for category in ["overall", *categories]:
        if category == "overall":
            rates = [float(row["rejection_rate"]) for row in audited]
            pooled = pooled_rejection
        else:
            rates = [
                int(row["rejection_counts"].get(category, 0))
                / max(1, int(row["samples_examined"]))
                for row in audited
            ]
            pooled = sum(
                int(row["rejection_counts"].get(category, 0)) for row in audited
            ) / max(1, sum(int(row["samples_examined"]) for row in audited))
        max_deviation = max(abs(value - pooled) for value in rates)
        flagged = max_deviation > 0.01
        category_rows[category] = {
            "pooled_rate": pooled,
            "maximum_absolute_shard_deviation": max_deviation,
            "coefficient_of_variation": (
                statistics.pstdev(rates) / statistics.fmean(rates)
                if statistics.fmean(rates)
                else 0.0
            ),
            "flagged": flagged,
        }
        if flagged:
            flags.append({"kind": "rejection_rate", "category": category})
    return {
        "status": "COMPLETE",
        "kind": "descriptive_homogeneity_check_not_hypothesis_test",
        "deterministic_shards": [row["filename"] for row in audited],
        "shard_count": len(audited),
        "rule": {
            "distribution": "range across shard statistics > 10% pooled mean",
            "rejection": "any shard differs from pooled rate by > 1 percentage point",
        },
        "raw_per_shard": audited,
        "distribution_statistics": statistics_rows,
        "rejection_statistics": category_rows,
        "flags": flags,
        "empirically_benign": not flags,
    }


def finalise(pipeline: dict[str, Any]) -> dict[str, Any]:
    preflight = _preflight(pipeline)
    shard_count = int(pipeline["acquisition"]["download_train_shards"])
    summaries: list[dict[str, Any]] = []
    for index in range(shard_count):
        path = _root(pipeline) / f"manifests/inspect_{index:03d}.json"
        if not path.is_file():
            raise RuntimeError(f"inspection incomplete at seeded shard index {index}")
        summaries.append(json.loads(path.read_text()))
    bare_key_locations: dict[str, str] = {}
    duplicate_bare_keys: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    for summary in summaries:
        ledger = _path(summary["ledger"])
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["key"] in bare_key_locations:
                    duplicate_bare_keys.append(
                        {
                            "key": row["key"],
                            "first_shard": bare_key_locations[row["key"]],
                            "second_shard": row["shard"],
                        }
                    )
                else:
                    bare_key_locations[row["key"]] = row["shard"]
                if row["status"] == "USABLE":
                    candidates.append(row)
    raw_exclusions, pixel_exclusions, exclusion_manifest = _exclusion_hashes(pipeline)
    candidates.sort(key=lambda row: row["canonical_id"])
    random.Random(int(pipeline["acquisition"]["shuffle_seed"])).shuffle(candidates)
    raw_seen: set[str] = set()
    pixel_seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    removed = Counter()
    target = int(pipeline["acquisition"]["target_usable_pairs"])
    for row in candidates:
        if row["raw_sha256"] in raw_exclusions:
            removed["overlap_coco_or_flickr_raw_hash"] += 1
        elif row["pixel_sha256"] in pixel_exclusions:
            removed["overlap_coco_or_flickr_pixel_hash"] += 1
        elif row["raw_sha256"] in raw_seen:
            removed["duplicate_raw_hash"] += 1
        elif row["pixel_sha256"] in pixel_seen:
            removed["duplicate_pixel_hash"] += 1
        else:
            raw_seen.add(row["raw_sha256"])
            pixel_seen.add(row["pixel_sha256"])
            if len(selected) < target:
                selected.append(row)
            else:
                removed["usable_but_not_selected"] += 1
    if len(selected) < target:
        raise RuntimeError(
            f"298 shards yielded {len(selected)} valid unique pairs; target is {target}"
        )
    selected_path = _root(pipeline) / "manifests/selected_keys.jsonl"
    temporary = selected_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(selected):
            payload = {
                "position": position,
                "canonical_id": row["canonical_id"],
                "shard": row["shard"],
                "key": row["key"],
                "caption": row["caption"],
                "raw_sha256": row["raw_sha256"],
                "pixel_sha256": row["pixel_sha256"],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, selected_path)

    prefix_raw: set[str] = set()
    prefix_pixel: set[str] = set()
    prefix_count = 0
    smallest_prefix: int | None = None
    by_seed_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_seed_order[int(row["seed_order"])].append(row)
    for seed_order in range(shard_count):
        for row in by_seed_order[seed_order]:
            if (
                row["raw_sha256"] in raw_exclusions
                or row["pixel_sha256"] in pixel_exclusions
                or row["raw_sha256"] in prefix_raw
                or row["pixel_sha256"] in prefix_pixel
            ):
                continue
            prefix_raw.add(row["raw_sha256"])
            prefix_pixel.add(row["pixel_sha256"])
            prefix_count += 1
        if prefix_count >= target:
            smallest_prefix = seed_order + 1
            break
    if smallest_prefix is None:
        raise AssertionError("full selected pool reached target but seeded prefix did not")

    download_rows = []
    digest = hashlib.sha256()
    for shard in preflight["selected_shards"]:
        path = _root(pipeline) / "shards/train" / shard["filename"]
        row = {
            "seed_order": shard["seed_order"],
            "filename": shard["filename"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        download_rows.append(row)
        digest.update(
            f"{row['seed_order']}\t{row['filename']}\t{row['sha256']}\n".encode()
        )
    heterogeneity = _heterogeneity(pipeline, summaries)
    atomic_json(
        heterogeneity,
        _root(pipeline) / "manifests/shard_heterogeneity.json",
    )
    result = {
        "status": "COMPLETE",
        "repo_id": pipeline["acquisition"]["repo_id"],
        "revision": pipeline["acquisition"]["revision"],
        "downloaded_shards": shard_count,
        "downloaded_bytes": sum(row["bytes"] for row in download_rows),
        "downloaded_shard_manifest": download_rows,
        "aggregate_ordered_shard_sha256": digest.hexdigest(),
        "not_downloaded_shards": preflight["not_downloaded_shards"],
        "target_usable_pairs": target,
        "selected_key_manifest": str(selected_path.relative_to(ROOT)),
        "selected_key_manifest_sha256": sha256_file(selected_path),
        "candidate_usable_before_global_dedup": len(candidates),
        "deduplication_removed": dict(removed),
        "deduplication_against": exclusion_manifest,
        "bare_keys_globally_unique": not duplicate_bare_keys,
        "duplicate_bare_key_count": len(duplicate_bare_keys),
        "duplicate_bare_key_examples": duplicate_bare_keys[:100],
        "canonical_identifier": "(shard filename, sample key)",
        "smallest_sufficient_seeded_shard_prefix": smallest_prefix,
        "unused_shards_by_approved_definition": shard_count - smallest_prefix,
        "minimum_resolution_rejections": sum(
            row["minimum_resolution_rejections"] for row in summaries
        ),
        "heterogeneity_manifest": "data/cc3m_v1_1/manifests/shard_heterogeneity.json",
        "heterogeneity_flagged": bool(heterogeneity["flags"]),
        "near_duplicate_limitation": (
            "Exact hashes do not detect resized, recompressed, cropped, or "
            "otherwise transformed near-duplicates."
        ),
        "provenance_limitation": preflight["provenance_limitation"],
    }
    atomic_json(result, _root(pipeline) / "manifests/acquisition_complete.json")
    return result


def retention_refresh(pipeline: dict[str, Any]) -> dict[str, Any]:
    spec = pipeline["retention"]
    now_ns = time.time_ns()
    tar_rows = []
    for path in sorted((_root(pipeline) / "shards/train").glob("*.tar")):
        stat = path.stat()
        os.utime(path, ns=(now_ns, stat.st_mtime_ns), follow_symlinks=False)
        tar_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "atime_ns_after": path.stat().st_atime_ns,
            }
        )
    metadata: set[Path] = set()
    for pattern in spec["refresh_metadata_globs"]:
        metadata.update(path for path in ROOT.glob(str(pattern)) if path.is_file())
    for path in sorted(metadata):
        stat = path.stat()
        os.utime(path, ns=(now_ns, stat.st_mtime_ns), follow_symlinks=False)
    result = {
        "status": "COMPLETE",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "method": "os.utime(atime=now, mtime=preserved); stat-only does not refresh atime",
        "all_downloaded_tars_refreshed": True,
        "tar_count": len(tar_rows),
        "tar_samples": tar_rows,
        "metadata_files_refreshed": len(metadata),
    }
    _append_jsonl(_root(pipeline) / "ledgers/retention_runs.jsonl", result)
    atomic_json(result, _root(pipeline) / "manifests/retention_latest.json")
    return result


def retention_validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    if not bool(pipeline["retention"].get("refresh_all_downloaded_tars")):
        raise RuntimeError("retention must refresh every downloaded tar")
    if int(pipeline["retention"]["interval_days"]) >= int(
        pipeline["retention"]["policy_warning_days"]
    ):
        raise RuntimeError("retention interval does not precede warning threshold")
    script = ROOT / "slurm/cc3m_scale/retention.sbatch"
    if not script.is_file():
        raise FileNotFoundError(script)
    return {
        "status": "READY",
        "interval_days": pipeline["retention"]["interval_days"],
        "refresh_all_downloaded_tars": True,
        "retention_sbatch": str(script.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "mirror-preflight",
            "download-shard",
            "inspect-shard",
            "finalise",
            "retention-refresh",
            "retention-validate",
        ],
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    pipeline = _pipeline(args.pipeline)
    if args.command == "mirror-preflight":
        result = mirror_preflight(pipeline)
    elif args.command == "download-shard":
        result = download_shard(pipeline, args.index)
    elif args.command == "inspect-shard":
        result = inspect_shard(pipeline, args.index)
    elif args.command == "finalise":
        result = finalise(pipeline)
    elif args.command == "retention-refresh":
        result = retention_refresh(pipeline)
    else:
        result = retention_validate(pipeline)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
