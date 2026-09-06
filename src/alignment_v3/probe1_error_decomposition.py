from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from src.alignment_v3.fingerprint import read_fingerprint, runtime_versions, sha256_file
from src.alignment_v3.model import build_model
from src.alignment_v3.references import REFERENCE_CHECKPOINTS, build_reference, native_loader
from src.alignment_v3.training import load_training_checkpoint
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_csv, atomic_json
from src.utils.config import deep_update, load_config
from src.utils.device import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = "configs/probe1_error_decomposition/pipeline.yaml"
PARTITIONS = ("rank_1", "rank_2_10", "outside_top_10")
BINS = ("low", "medium", "high")
FAILURE_CATEGORIES = (
    "fine_grained_attribute",
    "counting_quantity",
    "spatial_relation",
    "action_interaction",
    "object_category_confusion",
    "scene_context_confusion",
    "text_ocr_content",
    "annotation_ambiguity",
    "other",
)
AUDIT_RUBRIC = (
    "annotated_image_uniquely_correct",
    "retrieved_alternative_equally_plausible",
    "retrieved_alternative_related_but_incorrect",
    "query_intrinsically_ambiguous_or_generic",
    "insufficient_evidence",
)


def _load_pipeline(path: str | Path = DEFAULT_PIPELINE) -> dict[str, Any]:
    value = Path(path)
    return load_config(value if value.is_absolute() else ROOT / value)


def _out(pipeline: dict[str, Any]) -> Path:
    return ROOT / str(pipeline["output_root"])


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _test_seal_guard(pipeline: dict[str, Any]) -> None:
    gallery = pipeline["gallery"]
    csv_path = str(gallery["csv"])
    sealed_csv = str(gallery["sealed_test_csv"])
    sealed_id = str(gallery["sealed_test_dataset_id"])
    if Path(csv_path) == Path(sealed_csv) or sealed_id.lower() in csv_path.lower():
        raise RuntimeError("Flickr TEST is sealed; Probe 1 accepts validation only")
    if "test.csv" in Path(csv_path).name.lower():
        raise RuntimeError("Flickr TEST-like input path rejected")


def _rows_from_manifest(path: Path) -> list[dict[str, Any]]:
    payload = _json(path)
    return list(payload["captions"])


def prepare_gallery(pipeline: dict[str, Any]) -> dict[str, Any]:
    _test_seal_guard(pipeline)
    spec = pipeline["gallery"]
    source = ROOT / str(spec["csv"])
    canonical = ROOT / str(spec["canonical_manifest"])
    if not source.is_file() or not canonical.is_file():
        raise FileNotFoundError("canonical Flickr validation inputs are missing")
    source_hash = sha256_file(source)
    if source_hash != str(spec["expected_csv_sha256"]):
        raise RuntimeError(f"Flickr validation CSV hash mismatch: {source_hash}")
    provenance = _json(canonical)
    if provenance.get("csv_sha256") != source_hash:
        raise RuntimeError("canonical Flickr validation manifest disagrees with CSV")
    if provenance.get("test_overlap_images") != 0 or provenance.get("test_sealed") is not True:
        raise RuntimeError("canonical validation/test-seal provenance is invalid")
    image_hashes = {str(row["path"]): str(row["sha256"]) for row in provenance["image_hashes"]}
    captions: list[dict[str, Any]] = []
    image_order: list[str] = []
    counts: Counter[str] = Counter()
    with source.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            image_path = str(row["image_path"])
            image_id = Path(image_path).stem
            if image_id not in image_order:
                image_order.append(image_id)
            ordinal = counts[image_id]
            counts[image_id] += 1
            text = str(row["caption"])
            captions.append(
                {
                    "row_index": row_index,
                    "caption_id": f"{image_id}::caption_{ordinal}",
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption_ordinal": ordinal,
                    "caption": text,
                    "caption_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    expected_images = int(spec["expected_images"])
    expected_captions = int(spec["expected_captions"])
    per_image = int(spec["captions_per_image"])
    if len(image_order) != expected_images or len(captions) != expected_captions:
        raise RuntimeError("Flickr validation gallery cardinality mismatch")
    if set(counts.values()) != {per_image}:
        raise RuntimeError(f"expected exactly {per_image} captions per image")
    if len({row["caption_id"] for row in captions}) != len(captions):
        raise RuntimeError("caption IDs are not unique")
    images = []
    by_id = {row["image_id"]: row["image_path"] for row in captions}
    for index, image_id in enumerate(image_order):
        path = by_id[image_id]
        absolute = ROOT / path
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        actual_hash = sha256_file(absolute)
        if image_hashes.get(path) != actual_hash:
            raise RuntimeError(f"image hash mismatch: {path}")
        images.append(
            {"gallery_index": index, "image_id": image_id, "image_path": path, "sha256": actual_hash}
        )
    payload = {
        "status": "FROZEN",
        "dataset": "Flickr30k Karpathy validation",
        "source_csv": str(spec["csv"]),
        "source_csv_sha256": source_hash,
        "canonical_manifest": str(spec["canonical_manifest"]),
        "canonical_manifest_sha256": sha256_file(canonical),
        "test_overlap_images": 0,
        "test_sealed": True,
        "images": images,
        "captions": captions,
    }
    destination = _out(pipeline) / "gallery_manifest.json"
    if destination.is_file() and _json(destination) != payload:
        raise RuntimeError("refusing to overwrite a different frozen gallery manifest")
    atomic_json(payload, destination)
    return {"status": "COMPLETE", "gallery_sha256": sha256_file(destination), "images": len(images), "captions": len(captions)}


def _top_other(similarity: np.ndarray, image_ids: list[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    n = similarity.shape[0]
    top_mean = np.empty(n, dtype=np.float64)
    top_max = np.empty(n, dtype=np.float64)
    ids = np.asarray(image_ids)
    for index in range(n):
        values = similarity[index].copy()
        values[ids == ids[index]] = -np.inf
        selected = np.partition(values, -k)[-k:]
        top_mean[index] = float(selected.mean())
        top_max[index] = float(selected.max())
    return top_mean, top_max


def _within_values(similarity: np.ndarray, image_ids: list[str]) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, image_id in enumerate(image_ids):
        grouped[image_id].append(index)
    result = {}
    for image_id, indices in grouped.items():
        if len(indices) != 5:
            raise RuntimeError(f"{image_id} does not have five captions")
        values = [similarity[a, b] for pos, a in enumerate(indices) for b in indices[pos + 1 :]]
        result[image_id] = float(np.mean(values))
    return result


def _bge_embeddings(captions: list[str], revision: str) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True)
    model = AutoModel.from_pretrained(name, revision=revision, local_files_only=True).eval()
    parts = []
    content_lengths = []
    with torch.inference_mode():
        for start in range(0, len(captions), 128):
            batch = tokenizer(captions[start : start + 128], padding=True, truncation=True, return_tensors="pt")
            output = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
            parts.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
            content_lengths.extend(batch["attention_mask"].sum(1).tolist())
    embeddings = torch.cat(parts).numpy().astype(np.float32, copy=False)
    return embeddings, {"content_token_lengths": _distribution(content_lengths)}


def _bin_edges(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return [float(np.quantile(array, 1 / 3)), float(np.quantile(array, 2 / 3))]


def _bin(value: float, edges: list[float]) -> str:
    if value <= edges[0]:
        return "low"
    if value <= edges[1]:
        return "medium"
    return "high"


def freeze_ambiguity(pipeline: dict[str, Any]) -> dict[str, Any]:
    out = _out(pipeline)
    gallery_path = out / "gallery_manifest.json"
    if not gallery_path.is_file():
        raise RuntimeError("gallery must be frozen before ambiguity")
    ambiguity_path = out / "ambiguity_bins.json"
    receipt_path = out / "ambiguity_freeze_receipt.json"
    if ambiguity_path.is_file() or receipt_path.is_file():
        if not (ambiguity_path.is_file() and receipt_path.is_file()):
            raise RuntimeError("partial ambiguity freeze exists; refusing to overwrite it")
        verified = verify_freeze(pipeline)
        return {"status": "SKIPPED_VERIFIED", **verified}
    forbidden = list((out / "ranks").glob("*")) + list((out / "embeddings").glob("*"))
    if forbidden or (out / "per_query.csv").exists() or (out / "report.json").exists():
        raise RuntimeError("ambiguity must be frozen before retrieval artifacts exist")
    rows = _rows_from_manifest(gallery_path)
    captions = [str(row["caption"]) for row in rows]
    image_ids = [str(row["image_id"]) for row in rows]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), norm="l2")
    tfidf = vectorizer.fit_transform(captions)
    tfidf_similarity = cosine_similarity(tfidf, dense_output=True).astype(np.float32)
    bge, bge_meta = _bge_embeddings(captions, str(pipeline["ambiguity"]["robustness_revision"]))
    bge_similarity = np.empty((len(rows), len(rows)), dtype=np.float32)
    for start in range(0, len(rows), 512):
        bge_similarity[start : start + 512] = bge[start : start + 512] @ bge.T

    payload: dict[str, Any] = {
        "status": "FROZEN_BEFORE_RETRIEVAL_JOIN",
        "gallery_manifest_sha256": sha256_file(gallery_path),
        "primary_metric": {
            "id": "tfidf_word_unigram_bigram_cosine",
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "cross_image_primary": "mean_top5",
            "cross_image_secondary": "maximum",
        },
        "robustness_metric": {
            "id": "BAAI/bge-small-en-v1.5",
            "revision": str(pipeline["ambiguity"]["robustness_revision"]),
            "pooling": "attention_mask_mean",
            "query_prefix": None,
            "normalization": "l2",
            **bge_meta,
        },
        "limitations": list(pipeline.get("limitations", [])),
        "values": [],
    }
    metric_values = {}
    for name, similarity in (("tfidf", tfidf_similarity), ("bge", bge_similarity)):
        within = _within_values(similarity, image_ids)
        top5, maximum = _top_other(similarity, image_ids)
        metric_values[name] = {"within": within, "top5": top5, "maximum": maximum}
        payload[f"{name}_edges"] = {
            "within_image_consistency": _bin_edges(within[row["image_id"]] for row in rows),
            "cross_image_ambiguity_mean_top5": _bin_edges(top5),
        }
    for index, row in enumerate(rows):
        value = {"caption_id": row["caption_id"], "image_id": row["image_id"]}
        for name in ("tfidf", "bge"):
            within = float(metric_values[name]["within"][row["image_id"]])
            top5 = float(metric_values[name]["top5"][index])
            maximum = float(metric_values[name]["maximum"][index])
            value[name] = {
                "within_image_consistency": within,
                "within_bin": _bin(within, payload[f"{name}_edges"]["within_image_consistency"]),
                "cross_image_ambiguity_mean_top5": top5,
                "cross_image_ambiguity_max": maximum,
                "cross_bin": _bin(top5, payload[f"{name}_edges"]["cross_image_ambiguity_mean_top5"]),
            }
        payload["values"].append(value)
    primary_bins = [row["tfidf"]["cross_bin"] for row in payload["values"]]
    robust_bins = [row["bge"]["cross_bin"] for row in payload["values"]]
    payload["metric_agreement"] = {
        "cross_axis_spearman": float(pd.Series([row["tfidf"]["cross_image_ambiguity_mean_top5"] for row in payload["values"]]).corr(pd.Series([row["bge"]["cross_image_ambiguity_mean_top5"] for row in payload["values"]]), method="spearman")),
        "cross_bin_exact_agreement": float(np.mean(np.asarray(primary_bins) == np.asarray(robust_bins))),
        "cross_bin_weighted_kappa": float(cohen_kappa_score(primary_bins, robust_bins, labels=list(BINS), weights="linear")),
        "cross_bin_confusion": pd.crosstab(pd.Series(primary_bins, name="tfidf"), pd.Series(robust_bins, name="bge")).reindex(index=BINS, columns=BINS, fill_value=0).to_dict(),
    }
    atomic_json(payload, ambiguity_path)
    receipt = {
        "status": "FROZEN",
        "written_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": _git_sha(),
        "gallery_manifest_sha256": sha256_file(gallery_path),
        "ambiguity_bins_sha256": sha256_file(ambiguity_path),
        "retrieval_artifacts_present_at_freeze": False,
    }
    atomic_json(receipt, out / "ambiguity_freeze_receipt.json")
    return {"status": "COMPLETE", **receipt, "captions": len(rows)}


def verify_freeze(pipeline: dict[str, Any]) -> dict[str, Any]:
    out = _out(pipeline)
    receipt = _json(out / "ambiguity_freeze_receipt.json")
    actual_gallery = sha256_file(out / "gallery_manifest.json")
    actual_ambiguity = sha256_file(out / "ambiguity_bins.json")
    if receipt.get("gallery_manifest_sha256") != actual_gallery:
        raise RuntimeError("gallery changed after ambiguity freeze")
    if receipt.get("ambiguity_bins_sha256") != actual_ambiguity:
        raise RuntimeError("ambiguity bins changed after freeze")
    if receipt.get("retrieval_artifacts_present_at_freeze") is not False:
        raise RuntimeError("invalid ambiguity freeze ordering receipt")
    return {"status": "PASS", "gallery_manifest_sha256": actual_gallery, "ambiguity_bins_sha256": actual_ambiguity}


def extraction_jobs(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        {"model": "c4", "seed": int(seed), "epoch": int(epoch), "id": f"c4_seed_{seed}"}
        for seed, epoch in pipeline["c4"]["selected_epochs"].items()
    ]
    jobs.append({"model": "openclip", "seed": None, "epoch": None, "id": "openclip"})
    return jobs


def _select_job(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    value = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else int(index)
    jobs = extraction_jobs(pipeline)
    if value < 0 or value >= len(jobs):
        raise IndexError(value)
    return jobs[value]


def _student_model_loader(pipeline: dict[str, Any], job: dict[str, Any], device: torch.device):
    root = ROOT / str(pipeline["c4"]["checkpoint_root"]) / f"distill_strength_1p0__seed_{job['seed']}"
    config = load_config(root / "config.yaml")
    if int(config["data"]["image_size"]) != 224:
        raise RuntimeError("C4 Probe 1 requires the selected 224px model")
    model = build_model(config).to(device).eval()
    fingerprint = read_fingerprint(root / "fingerprint.json")
    if fingerprint is None:
        raise RuntimeError("C4 fingerprint missing")
    checkpoint = root / f"epoch_{int(job['epoch']):02d}.pt"
    load_training_checkpoint(checkpoint, model, device=device, expected_fingerprint=fingerprint)
    eval_config = deep_update(
        config,
        {
            "data": {
                "val_csv": str(ROOT / str(pipeline["gallery"]["csv"])),
                "image_root": str(ROOT),
                "num_workers": int(pipeline["evaluation"]["num_workers"]),
                "persistent_workers": False,
            },
            "training": {"batch_size": int(pipeline["evaluation"]["batch_size"])},
        },
    )
    _, loader = build_dataloaders(eval_config)
    return model, loader, {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint": fingerprint.digest,
        "image_resolution": 224,
        "image_preprocessing": {
            "image_size": int(config["data"]["image_size"]),
            "mean": config["data"].get("image_mean"),
            "std": config["data"].get("image_std"),
            "interpolation": str(config["data"].get("interpolation", "bicubic")),
            "source": "selected C4 checkpoint config.yaml",
        },
        "tokenizer_policy": "dynamic_padding_per_batch",
    }


def _reference_model_loader(pipeline: dict[str, Any], device: torch.device):
    spec = pipeline["openclip"]
    reference_id = str(spec["reference_id"])
    expected = (str(spec["model_name"]), str(spec["pretrained"]))
    if REFERENCE_CHECKPOINTS.get(reference_id) != expected:
        raise RuntimeError("OpenCLIP registry identity differs from frontier control")
    model = build_reference(reference_id).to(device).eval()
    loader = native_loader(
        model,
        ROOT / str(pipeline["gallery"]["csv"]),
        image_root=ROOT,
        batch_size=int(pipeline["evaluation"]["batch_size"]),
        num_workers=int(pipeline["evaluation"]["num_workers"]),
    )
    return model, loader, {
        "reference_id": reference_id,
        "model_name": model.model_name,
        "pretrained": model.pretrained,
        "image_resolution": model.image_size,
        "image_preprocessing": {"transform": repr(model.preprocess), "source": "checkpoint-native open_clip transform"},
        "tokenizer_policy": "checkpoint_native_fixed_context",
        "native_context_length": model.native_context_length,
        "expected_weight_sha256": str(spec["weight_sha256"]),
    }


def _distribution(values: Iterable[int | float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size), "min": float(array.min()), "p5": float(np.quantile(array, .05)),
        "median": float(np.median(array)), "mean": float(array.mean()), "sd": float(array.std(ddof=0)),
        "p95": float(np.quantile(array, .95)), "max": float(array.max()),
    }


@torch.no_grad()
def _extract_once(model: Any, loader: Any, device: torch.device) -> dict[str, Any]:
    images, texts, image_paths, text_paths, captions = [], [], [], [], []
    content_lengths, batch_widths = [], []
    for batch in loader:
        batch_images = batch["images"].to(device)
        batch_captions = list(batch["captions"])
        tokenizer_owner = model.text_encoder if hasattr(model, "text_encoder") else model
        tokens = tokenizer_owner.tokenize(batch_captions)
        if isinstance(tokens, dict) or hasattr(tokens, "keys"):
            mask = tokens["attention_mask"]
            content_lengths.extend(int(value) for value in mask.sum(1).tolist())
            batch_widths.extend([int(mask.shape[1])] * len(batch_captions))
        else:
            content_lengths.extend(int(value) for value in tokens.ne(0).sum(1).tolist())
            batch_widths.extend([int(tokens.shape[1])] * len(batch_captions))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            images.append(model.encode_image(batch_images).detach().float().cpu())
            texts.append(model.encode_text_tokens(tokens).detach().float().cpu())
        image_paths.extend(str(value) for value in batch["image_paths"])
        text_paths.extend(str(value) for value in batch["text_image_paths"])
        captions.extend(batch_captions)
    return {
        "image_embeds": torch.cat(images), "text_embeds": torch.cat(texts),
        "image_paths": image_paths, "text_image_paths": text_paths, "captions": captions,
        "content_token_lengths": content_lengths, "padded_batch_widths": batch_widths,
    }


def extract(pipeline: dict[str, Any], index: int | None) -> dict[str, Any]:
    _test_seal_guard(pipeline)
    verify_freeze(pipeline)
    job = _select_job(pipeline, index)
    device = get_device("auto")
    if device.type != "cuda":
        raise RuntimeError("Probe 1 extraction requires CUDA")
    gpu = torch.cuda.get_device_name(device)
    if "RTX PRO 6000" not in gpu or "Blackwell" not in gpu or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"Probe 1 requires teaching/native-BF16 Blackwell, got {gpu}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if job["model"] == "c4":
        model, loader, provenance = _student_model_loader(pipeline, job, device)
    else:
        model, loader, provenance = _reference_model_loader(pipeline, device)
    first = _extract_once(model, loader, device)
    second = _extract_once(model, loader, device)
    destination = _out(pipeline) / "embeddings"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(first, destination / f"{job['id']}.pt")
    torch.save(second, destination / f"{job['id']}.repeat.pt")
    metadata = {
        "status": "COMPLETE", **job, **provenance,
        "gpu_model": gpu, "node": platform.node(), "precision": "native_bf16",
        "content_token_lengths": _distribution(first["content_token_lengths"]),
        "padded_batch_widths": _distribution(first["padded_batch_widths"]),
        "fixed_vs_dynamic_tokenizer_asymmetry": "OpenCLIP fixed context and MiniLM dynamic padding are checkpoint-native; no latency is compared in Probe 1.",
    }
    atomic_json(metadata, destination / f"{job['id']}.json")
    return metadata


def _stable_order(similarity: torch.Tensor) -> torch.Tensor:
    return torch.argsort(similarity, dim=1, descending=True, stable=True)


def rank_payload(embedding: dict[str, Any], gallery: dict[str, Any], model_id: str, seed: int | None) -> pd.DataFrame:
    image = torch.nn.functional.normalize(torch.as_tensor(embedding["image_embeds"]).float(), dim=-1)
    text = torch.nn.functional.normalize(torch.as_tensor(embedding["text_embeds"]).float(), dim=-1)
    image_ids = [str(row["image_id"]) for row in gallery["images"]]
    caption_ids = [str(row["caption_id"]) for row in gallery["captions"]]
    caption_image_ids = [str(row["image_id"]) for row in gallery["captions"]]
    if len(image_ids) != image.shape[0] or len(caption_ids) != text.shape[0]:
        raise RuntimeError("embedding/gallery cardinality mismatch")
    if [Path(value).stem for value in embedding["image_paths"]] != image_ids:
        raise RuntimeError("image embedding order differs from frozen gallery")
    rows = []
    i2t_order = _stable_order(image @ text.T)
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, image_id in enumerate(caption_image_ids):
        by_image[image_id].append(index)
    for query_index, image_id in enumerate(image_ids):
        order = i2t_order[query_index].tolist()
        inverse = {target: rank + 1 for rank, target in enumerate(order)}
        positives = by_image[image_id]
        ranks = [inverse[index] for index in positives]
        best_pos = int(np.argmin(ranks))
        rows.append({
            "model": model_id, "seed": seed, "direction": "i2t", "query_id": image_id,
            "positive_id": caption_ids[positives[best_pos]], "positive_rank": min(ranks),
            "all_positive_ids": json.dumps([caption_ids[index] for index in positives]),
            "all_positive_ranks": json.dumps(ranks),
            "top10_retrieved_ids": json.dumps([caption_ids[index] for index in order[:10]]),
        })
    t2i_order = _stable_order(text @ image.T)
    image_index = {value: index for index, value in enumerate(image_ids)}
    for query_index, caption_id in enumerate(caption_ids):
        order = t2i_order[query_index].tolist()
        positive = image_index[caption_image_ids[query_index]]
        rank = order.index(positive) + 1
        rows.append({
            "model": model_id, "seed": seed, "direction": "t2i", "query_id": caption_id,
            "positive_id": image_ids[positive], "positive_rank": rank,
            "all_positive_ids": json.dumps([image_ids[positive]]), "all_positive_ranks": json.dumps([rank]),
            "top10_retrieved_ids": json.dumps([image_ids[index] for index in order[:10]]),
        })
    frame = pd.DataFrame(rows)
    frame["partition"] = np.select(
        [frame.positive_rank.eq(1), frame.positive_rank.le(10)],
        ["rank_1", "rank_2_10"], default="outside_top_10",
    )
    return frame


def rank(pipeline: dict[str, Any]) -> dict[str, Any]:
    verify_freeze(pipeline)
    out = _out(pipeline)
    gallery = _json(out / "gallery_manifest.json")
    destination = out / "ranks"
    destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    for job in extraction_jobs(pipeline):
        first = torch.load(out / "embeddings" / f"{job['id']}.pt", map_location="cpu", weights_only=False)
        repeat = torch.load(out / "embeddings" / f"{job['id']}.repeat.pt", map_location="cpu", weights_only=False)
        first_frame = rank_payload(first, gallery, job["id"], job["seed"])
        repeat_frame = rank_payload(repeat, gallery, job["id"], job["seed"])
        compare = ["direction", "query_id", "positive_rank", "top10_retrieved_ids"]
        if not first_frame[compare].equals(repeat_frame[compare]):
            unequal = first_frame[compare].ne(repeat_frame[compare]).any(axis=1)
            row = int(np.flatnonzero(unequal.to_numpy())[0])
            raise RuntimeError(f"determinism failure for {job['id']} at row {row}")
        atomic_csv(first_frame, destination / f"{job['id']}.csv")
        receipts.append({"id": job["id"], "rows": len(first_frame), "rank_sha256": sha256_file(destination / f"{job['id']}.csv"), "repeat_identical": True})
    payload = {"status": "PASS", "tie_break": pipeline["ranking"]["tie_break"], "models": receipts}
    atomic_json(payload, out / "determinism.json")
    return payload


def _metric_payload(frame: pd.DataFrame) -> dict[str, Any]:
    ranks = frame.positive_rank.to_numpy(dtype=np.int64)
    result: dict[str, Any] = {
        "queries": int(len(ranks)), "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1.0 / ranks)),
        "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
        "outside_top10_fraction": float(np.mean(ranks > 10)),
    }
    result["ranking_efficiency_R1_over_R10"] = result["R@1"] / result["R@10"] if result["R@10"] else None
    counts = {str(rank): int(np.sum(ranks == rank)) for rank in range(1, 11)}
    result["rank_1_10_counts"] = counts
    result["rank_1_10_histogram_all_queries"] = {key: value / len(ranks) for key, value in counts.items()}
    partition = frame.partition.value_counts().reindex(PARTITIONS, fill_value=0)
    result["failure_partition"] = {key: {"count": int(partition[key]), "proportion": float(partition[key] / len(frame))} for key in PARTITIONS}
    return result


def _intersection(c4: pd.DataFrame, control: pd.DataFrame) -> dict[str, Any]:
    left = c4.set_index("query_id")
    right = control.set_index("query_id")
    common = left.index.intersection(right.index)
    a = left.loc[common].positive_rank.le(10)
    b = right.loc[common].positive_rank.le(10)
    both = a & b
    result = {
        "intersection_size": int(both.sum()), "c4_exclusive_size": int((a & ~b).sum()),
        "openclip_exclusive_size": int((~a & b).sum()), "neither_size": int((~a & ~b).sum()),
    }
    for label, frame in (("c4", left.loc[common][both]), ("openclip", right.loc[common][both])):
        ranks = frame.positive_rank.to_numpy(int)
        result[label] = {
            "rank1_given_intersection": float(np.mean(ranks == 1)) if len(ranks) else None,
            "rank_1_10_counts": {str(rank): int(np.sum(ranks == rank)) for rank in range(1, 11)},
            "rank_1_10_histogram": {str(rank): float(np.mean(ranks == rank)) for rank in range(1, 11)},
        }
    return result


def _bootstrap_intersection_difference(
    c4: pd.DataFrame,
    control: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    percent: float,
) -> dict[str, Any]:
    left = c4.set_index("query_id")
    right = control.set_index("query_id")
    query_ids = left.index.intersection(right.index).to_numpy()
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    intersection_sizes: list[int] = []
    for _ in range(replicates):
        sampled = rng.choice(query_ids, size=len(query_ids), replace=True)
        left_ranks = left.loc[sampled].positive_rank.to_numpy(dtype=np.int64)
        right_ranks = right.loc[sampled].positive_rank.to_numpy(dtype=np.int64)
        intersection = (left_ranks <= 10) & (right_ranks <= 10)
        intersection_sizes.append(int(intersection.sum()))
        if not intersection.any():
            continue
        differences.append(
            float(np.mean(left_ranks[intersection] == 1) - np.mean(right_ranks[intersection] == 1))
        )
    if len(differences) != replicates:
        raise RuntimeError("empty top-10 intersection in a bootstrap replicate")
    return {
        "statistic": "C4 minus OpenCLIP P(rank=1 | both rank<=10)",
        "replicates": replicates,
        "percentile_interval": _percentile_interval(differences, percent),
        "bootstrap_mean": float(np.mean(differences)),
        "intersection_size_range": [int(min(intersection_sizes)), int(max(intersection_sizes))],
    }


def _percentile_interval(values: list[float], percent: float) -> list[float]:
    tail = (100.0 - percent) / 2.0
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))]


def _bootstrap_excess(c4: dict[str, pd.DataFrame], control: dict[str, pd.DataFrame], pipeline: dict[str, Any]) -> dict[str, Any]:
    gallery = _json(_out(pipeline) / "gallery_manifest.json")
    image_ids = [row["image_id"] for row in gallery["images"]]
    captions_by_image: dict[str, list[str]] = defaultdict(list)
    for row in gallery["captions"]:
        captions_by_image[row["image_id"]].append(row["caption_id"])
    rng = np.random.default_rng(int(pipeline["bootstrap"]["seed"]))
    values: dict[str, list[float]] = {seed: [] for seed in c4}
    for _ in range(int(pipeline["bootstrap"]["replicates"])):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True).tolist()
        caption_sample = [caption for image_id in sampled for caption in captions_by_image[image_id]]
        for seed in c4:
            ci = c4[seed]["i2t"].set_index("query_id").loc[sampled].positive_rank.to_numpy()
            ct = c4[seed]["t2i"].set_index("query_id").loc[caption_sample].positive_rank.to_numpy()
            oi = control["i2t"].set_index("query_id").loc[sampled].positive_rank.to_numpy()
            ot = control["t2i"].set_index("query_id").loc[caption_sample].positive_rank.to_numpy()
            values[seed].append(float(((ci <= 1).mean() - (ct <= 1).mean()) - ((oi <= 1).mean() - (ot <= 1).mean())))
    percent = float(pipeline["bootstrap"]["interval_percent"])
    return {seed: {"replicates": len(rows), "percentile_interval": _percentile_interval(rows, percent)} for seed, rows in values.items()}


def _attach_ambiguity(frame: pd.DataFrame, ambiguity: dict[str, Any], gallery: dict[str, Any]) -> pd.DataFrame:
    by_caption = {row["caption_id"]: row for row in ambiguity["values"]}
    captions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ambiguity["values"]:
        captions_by_image[row["image_id"]].append(row)
    values = []
    for row in frame.to_dict("records"):
        if row["direction"] == "t2i":
            source = by_caption[row["query_id"]]
        else:
            group = captions_by_image[row["query_id"]]
            source = {
                "tfidf": {"within_image_consistency": group[0]["tfidf"]["within_image_consistency"], "within_bin": group[0]["tfidf"]["within_bin"], "cross_image_ambiguity_mean_top5": float(np.mean([x["tfidf"]["cross_image_ambiguity_mean_top5"] for x in group])), "cross_bin": None},
                "bge": {"within_image_consistency": group[0]["bge"]["within_image_consistency"], "within_bin": group[0]["bge"]["within_bin"], "cross_image_ambiguity_mean_top5": float(np.mean([x["bge"]["cross_image_ambiguity_mean_top5"] for x in group])), "cross_bin": None},
            }
        row.update({
            "tfidf_within_consistency": source["tfidf"]["within_image_consistency"], "tfidf_within_bin": source["tfidf"]["within_bin"],
            "tfidf_cross_ambiguity": source["tfidf"]["cross_image_ambiguity_mean_top5"], "tfidf_cross_bin": source["tfidf"].get("cross_bin"),
            "bge_within_consistency": source["bge"]["within_image_consistency"], "bge_within_bin": source["bge"]["within_bin"],
            "bge_cross_ambiguity": source["bge"]["cross_image_ambiguity_mean_top5"], "bge_cross_bin": source["bge"].get("cross_bin"),
        })
        values.append(row)
    return pd.DataFrame(values)


def _t2i_bins(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    subset = frame[frame.direction.eq("t2i")]
    result = {"cross_axis": {}, "within_axis": {}, "two_dimensional": {}}
    for name, column in (("cross_axis", f"{metric}_cross_bin"), ("within_axis", f"{metric}_within_bin")):
        for value in BINS:
            rows = subset[subset[column].eq(value)]
            result[name][value] = {"queries": len(rows), "R@1": float(rows.positive_rank.le(1).mean()) if len(rows) else None}
    for within in BINS:
        result["two_dimensional"][within] = {}
        for cross in BINS:
            rows = subset[subset[f"{metric}_within_bin"].eq(within) & subset[f"{metric}_cross_bin"].eq(cross)]
            result["two_dimensional"][within][cross] = {"queries": len(rows), "R@1": float(rows.positive_rank.le(1).mean()) if len(rows) else None}
    return result


def _audit_export(all_rows: pd.DataFrame, gallery: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
    seed = int(pipeline["exports"]["display_seed"])
    source = all_rows[(all_rows.model.eq(f"c4_seed_{seed}")) & all_rows.direction.eq("t2i")].copy()
    populations = source.groupby(["partition", "tfidf_cross_bin"], dropna=False).size()
    population_payload = {f"{part}|{bin_name}": int(populations.get((part, bin_name), 0)) for part in PARTITIONS for bin_name in BINS}
    rng = np.random.default_rng(int(pipeline["exports"]["audit_seed"]))
    selected = []
    capacity = {}
    for part in PARTITIONS:
        for bin_name in BINS:
            rows = source[source.partition.eq(part) & source.tfidf_cross_bin.eq(bin_name)]
            order = rng.permutation(len(rows))
            take = min(11, len(rows))
            selected.extend(rows.iloc[order[:take]].to_dict("records"))
            capacity[(part, bin_name)] = rows.iloc[order[take:]].to_dict("records")
    target = int(pipeline["exports"]["audit_size"])
    while len(selected) < target:
        available = [(len(rows), part, bin_name) for (part, bin_name), rows in capacity.items() if rows]
        if not available:
            break
        _, part, bin_name = max(available, key=lambda value: (value[0], value[1], value[2]))
        selected.append(capacity[(part, bin_name)].pop(0))
    if len(selected) != target:
        raise RuntimeError(f"could not allocate {target} audit rows")
    image_path = {row["image_id"]: row["image_path"] for row in gallery["images"]}
    caption_text = {row["caption_id"]: row["caption"] for row in gallery["captions"]}
    blinded, key = [], []
    for number, row in enumerate(selected):
        retrieved = json.loads(row["top10_retrieved_ids"])[:5]
        candidates = list(dict.fromkeys(retrieved + [row["positive_id"]]))
        local_rng = random.Random(int(pipeline["exports"]["audit_seed"]) + number)
        local_rng.shuffle(candidates)
        labels = [chr(ord("A") + index) for index in range(len(candidates))]
        display = dict(zip(labels, [image_path[value] for value in candidates]))
        audit_id = f"audit_{number:03d}"
        blinded.append({"audit_id": audit_id, "query_caption": caption_text[row["query_id"]], **{f"candidate_{label}": display[label] for label in labels}, "rubric_categories": json.dumps(AUDIT_RUBRIC)})
        key.append({"audit_id": audit_id, "query_id": row["query_id"], "positive_id": row["positive_id"], "positive_rank": int(row["positive_rank"]), "partition": row["partition"], "tfidf_cross_bin": row["tfidf_cross_bin"], "bge_cross_bin": row["bge_cross_bin"], "candidate_mapping": json.dumps(dict(zip(labels, candidates)))})
    out = _out(pipeline)
    blinded_frame = pd.DataFrame(blinded)
    _assert_blinded_export(blinded_frame)
    atomic_csv(blinded_frame, out / "audit_sample_blinded.csv")
    atomic_csv(pd.DataFrame(key), out / "audit_key.csv")
    return {"population_counts_before_redistribution": population_payload, "sample_size": len(blinded), "sampling_rule": "11 from each partition x TF-IDF cross-ambiguity cell; deficit redistributed by remaining capacity; one final slot to largest remaining stratum"}


def _assert_blinded_export(frame: pd.DataFrame) -> None:
    forbidden = {
        "query_id", "positive_id", "positive_rank", "partition",
        "tfidf_cross_bin", "bge_cross_bin", "candidate_mapping",
    }
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise RuntimeError(f"blinded audit export leaks answer fields: {leaked}")


def _failure_export(all_rows: pd.DataFrame, gallery: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
    caption_text = {row["caption_id"]: row["caption"] for row in gallery["captions"]}
    image_path = {row["image_id"]: row["image_path"] for row in gallery["images"]}
    caption_image = {row["caption_id"]: row["image_id"] for row in gallery["captions"]}
    result = {}
    output_rows = []
    target = int(pipeline["exports"]["failure_size_per_direction"])
    rng = np.random.default_rng(int(pipeline["exports"]["failure_seed"]))
    c4 = all_rows[all_rows.model.str.startswith("c4_seed_")]
    for direction in ("i2t", "t2i"):
        subset = c4[c4.direction.eq(direction)]
        pivot = subset.pivot(index="query_id", columns="seed", values="positive_rank")
        consensus = pivot[(pivot > 10).all(axis=1)].index.tolist()
        result[direction] = {"all_three_seed_outside_top10_count": len(consensus), "exported": 0, "stopped_below_50": len(consensus) < target}
        if len(consensus) < target:
            continue
        selected = rng.choice(consensus, size=target, replace=False).tolist()
        display = subset[subset.seed.eq(int(pipeline["exports"]["display_seed"]))].set_index("query_id")
        for query_id in selected:
            row = display.loc[query_id]
            retrieved_ids = json.loads(row.top10_retrieved_ids)
            if direction == "t2i":
                retrieved_items = [{"image_id": value, "image_path": image_path[value]} for value in retrieved_ids]
                positive_item = {"image_id": row.positive_id, "image_path": image_path[row.positive_id]}
            else:
                retrieved_items = [
                    {"caption_id": value, "caption": caption_text[value], "image_id": caption_image[value]}
                    for value in retrieved_ids
                ]
                positive_item = {"caption_id": row.positive_id, "caption": caption_text[row.positive_id], "image_id": caption_image[row.positive_id]}
            output_rows.append({
                "direction": direction, "query_id": query_id,
                "query": caption_text[query_id] if direction == "t2i" else image_path[query_id],
                "positive_id": row.positive_id, "positive_rank_seed42": int(row.positive_rank),
                "positive_item": json.dumps(positive_item),
                "ranks_all_seeds": json.dumps({str(seed): int(pivot.loc[query_id, seed]) for seed in pivot.columns}),
                "top10_seed42": row.top10_retrieved_ids,
                "retrieved_items_seed42": json.dumps(retrieved_items),
                "predeclared_categories": json.dumps(FAILURE_CATEGORIES),
            })
        result[direction]["exported"] = target
    atomic_csv(pd.DataFrame(output_rows), _out(pipeline) / "failure_sample.csv")
    return result


def report(pipeline: dict[str, Any]) -> dict[str, Any]:
    freeze = verify_freeze(pipeline)  # Hard pre-join circularity gate.
    out = _out(pipeline)
    gallery = _json(out / "gallery_manifest.json")
    ambiguity = _json(out / "ambiguity_bins.json")
    frames = []
    for job in extraction_jobs(pipeline):
        frames.append(_attach_ambiguity(pd.read_csv(out / "ranks" / f"{job['id']}.csv"), ambiguity, gallery))
    all_rows = pd.concat(frames, ignore_index=True)
    atomic_csv(all_rows, out / "per_query.csv")
    models = {}
    split_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for model, model_frame in all_rows.groupby("model"):
        split_frames[model] = {}
        models[model] = {"directions": {}}
        for direction, direction_frame in model_frame.groupby("direction"):
            split_frames[model][direction] = direction_frame
            models[model]["directions"][direction] = _metric_payload(direction_frame)
        models[model]["raw_asymmetry_R1"] = models[model]["directions"]["i2t"]["R@1"] - models[model]["directions"]["t2i"]["R@1"]
        models[model]["ambiguity"] = {metric: _t2i_bins(model_frame, metric) for metric in ("tfidf", "bge")}
    control = split_frames["openclip"]
    intersections, excess = {}, {}
    c4_split = {}
    for seed in (42, 43, 44):
        model = f"c4_seed_{seed}"
        c4_split[str(seed)] = split_frames[model]
        excess[str(seed)] = models[model]["raw_asymmetry_R1"] - models["openclip"]["raw_asymmetry_R1"]
        intersections[str(seed)] = {}
        for direction_index, direction in enumerate(("i2t", "t2i")):
            value = _intersection(split_frames[model][direction], control[direction])
            value["paired_query_bootstrap"] = _bootstrap_intersection_difference(
                split_frames[model][direction],
                control[direction],
                replicates=int(pipeline["bootstrap"]["replicates"]),
                seed=int(pipeline["bootstrap"]["seed"]) + seed * 10 + direction_index,
                percent=float(pipeline["bootstrap"]["interval_percent"]),
            )
            intersections[str(seed)][direction] = value
    seed_metrics = {}
    for direction in ("i2t", "t2i"):
        seed_metrics[direction] = {}
        for metric in ("R@1", "R@5", "R@10", "MRR", "mean_rank", "median_rank"):
            values = [models[f"c4_seed_{seed}"]["directions"][direction][metric] for seed in (42, 43, 44)]
            seed_metrics[direction][metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "values": dict(zip(("42", "43", "44"), values))}
    audit = _audit_export(all_rows, gallery, pipeline)
    failures = _failure_export(all_rows, gallery, pipeline)
    payload = {
        "status": "COMPLETE", "diagnostic_only": True, "flickr_test_used": False,
        "freeze_gate_at_join": freeze,
        "interpretation_constraints": {
            "directional_asymmetry": "Flickr's five-caption structure inflates i2t relative to t2i for every dual encoder. Only excess over the matched OpenCLIP control can support a model-specific directional claim, not a component attribution.",
            "ambiguity": "This estimates an ambiguity-associated performance ceiling, not an absolute irrecoverable ceiling.",
            "bge_independence": "BGE is non-circular but shares broad data/objective lineage with sentence-transformer models and is not independent semantic ground truth; the human audit is the arbiter.",
            "tokenizers": "OpenCLIP's fixed 77-token context and MiniLM's dynamic padding are checkpoint-native. Probe 1 measures ranking, not latency; both rank the identical frozen gallery with cosine similarity and the same tie rule.",
        },
        "models": models, "c4_three_seed_summary": seed_metrics,
        "excess_asymmetry_by_seed": excess,
        "excess_asymmetry_cluster_bootstrap": _bootstrap_excess(c4_split, control, pipeline),
        "intersection_ranking_efficiency": intersections,
        "ambiguity_metric_agreement": ambiguity["metric_agreement"],
        "audit_export": audit, "failure_export": failures,
        "tie_break": pipeline["ranking"]["tie_break"],
        "determinism": _json(out / "determinism.json"),
    }
    atomic_json(payload, out / "report.json")
    environment = {
        "status": "COMPLETE", "git_sha": _git_sha(), "runtime_versions": runtime_versions(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__, "platform": platform.platform(),
        "gallery_manifest_sha256": sha256_file(out / "gallery_manifest.json"),
        "ambiguity_bins_sha256": sha256_file(out / "ambiguity_bins.json"),
        "ranking": pipeline["ranking"], "bootstrap": pipeline["bootstrap"],
        "random_seeds": {"bootstrap": pipeline["bootstrap"]["seed"], "audit": pipeline["exports"]["audit_seed"], "failure": pipeline["exports"]["failure_seed"]},
        "model_metadata": {job["id"]: _json(out / "embeddings" / f"{job['id']}.json") for job in extraction_jobs(pipeline)},
    }
    atomic_json(environment, out / "environment.json")
    return payload


def validate(pipeline: dict[str, Any]) -> dict[str, Any]:
    _test_seal_guard(pipeline)
    expected = (str(pipeline["openclip"]["model_name"]), str(pipeline["openclip"]["pretrained"]))
    if REFERENCE_CHECKPOINTS.get(str(pipeline["openclip"]["reference_id"])) != expected:
        raise RuntimeError("OpenCLIP registry mismatch")
    reference_manifest = ROOT / "results/efficiency_frontier/manifests/reference_weights/openclip_vit_b32_quickgelu_openai.json"
    if not reference_manifest.is_file():
        raise FileNotFoundError(reference_manifest)
    reference_payload = _json(reference_manifest)
    expected_weight = str(pipeline["openclip"]["weight_sha256"])
    recorded_weights = [str(row["sha256"]) for row in reference_payload.get("files", [])]
    if expected_weight not in recorded_weights:
        raise RuntimeError("OpenCLIP local weight checksum differs from frontier control")
    matching = [row for row in reference_payload.get("files", []) if str(row.get("sha256")) == expected_weight]
    if len(matching) != 1:
        raise RuntimeError("OpenCLIP frontier manifest does not identify exactly one expected weight file")
    weight_path = Path(str(matching[0]["path"]))
    if not weight_path.is_file() or sha256_file(weight_path) != expected_weight:
        raise RuntimeError("OpenCLIP weight file is absent or no longer matches its recorded checksum")
    for job in extraction_jobs(pipeline)[:-1]:
        root = ROOT / str(pipeline["c4"]["checkpoint_root"]) / f"distill_strength_1p0__seed_{job['seed']}"
        for path in (root / "config.yaml", root / "fingerprint.json", root / f"epoch_{job['epoch']:02d}.pt"):
            if not path.is_file():
                raise FileNotFoundError(path)
    bge = ROOT / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots" / str(pipeline["ambiguity"]["robustness_revision"])
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    if any(not (bge / name).is_file() for name in required):
        raise FileNotFoundError("complete pinned BGE cache is unavailable")
    return {"status": "READY", "extraction_jobs": extraction_jobs(pipeline), "flickr_test_sealed": True, "bge_revision": bge.name, "openclip_registry": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "prepare-gallery", "freeze-ambiguity", "extract", "rank", "report", "verify-freeze"))
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    pipeline = _load_pipeline(args.pipeline)
    actions = {
        "validate": lambda: validate(pipeline), "prepare-gallery": lambda: prepare_gallery(pipeline),
        "freeze-ambiguity": lambda: freeze_ambiguity(pipeline), "extract": lambda: extract(pipeline, args.index),
        "rank": lambda: rank(pipeline), "report": lambda: report(pipeline), "verify-freeze": lambda: verify_freeze(pipeline),
    }
    print(json.dumps(actions[args.command](), indent=2, default=str))


if __name__ == "__main__":
    main()
