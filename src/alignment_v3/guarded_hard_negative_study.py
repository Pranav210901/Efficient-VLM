from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from src.alignment_v3.distillation import _key
from src.alignment_v3.efficiency_frontier import _hardware_guard
from src.alignment_v3.fingerprint import read_fingerprint
from src.alignment_v3.model import build_model
from src.alignment_v3.runner import ROOT
from src.alignment_v3.training import load_training_checkpoint
from src.data import build_dataloaders
from src.phase15.io_utils import atomic_csv, atomic_json, sha256_file
from src.utils.config import load_config


CONFIG_ROOT = ROOT / "configs/guarded_hard_negative_study"
OUT = ROOT / "results/guarded_hard_negative_study/mining"
PREREG = CONFIG_ROOT / "preregistration.md"
CALIBRATION = CONFIG_ROOT / "threshold_calibration.json"
RECEIPT = CONFIG_ROOT / "freeze_receipt.json"
SEEDS = (42, 43, 44)
EPOCHS = {42: 22, 43: 22, 44: 20}
CHECKPOINT_ROOT = ROOT / "checkpoints/text_aggregation_study/M_T1/sensitivity"
CANDIDATE_DEPTH = 64
NEGATIVES_PER_OWNER = 8
IMAGE_TEXT_THRESHOLD = 0.222935
TEXT_TEXT_THRESHOLD = 0.765541
ENCODE_BATCH_SIZE = 256
MINE_BLOCK_SIZE = 64


def _sha_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _freeze_gate() -> dict[str, Any]:
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "FROZEN" or not receipt.get("approved_by_user"):
        raise RuntimeError("hard-negative preregistration is not user-approved and frozen")
    for relative, expected in receipt["files"].items():
        path = ROOT / relative
        actual = _sha_text(path)
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {relative}: {actual} != {expected}")
    if "**Status:** `FROZEN`" not in PREREG.read_text():
        raise RuntimeError("preregistration status is not FROZEN")
    calibration = json.loads(CALIBRATION.read_text())
    if not calibration.get("computed_before_hard_negative_mining"):
        raise RuntimeError("threshold calibration ordering is invalid")
    if float(calibration["teacher_image_text"]["frozen_threshold"]) != IMAGE_TEXT_THRESHOLD:
        raise RuntimeError("image-text threshold differs from frozen implementation constant")
    if float(calibration["teacher_text_text"]["frozen_threshold"]) != TEXT_TEXT_THRESHOLD:
        raise RuntimeError("text-text threshold differs from frozen implementation constant")
    return receipt


def _checkpoint_inputs(seed: int) -> tuple[Path, Path, Path]:
    directory = CHECKPOINT_ROOT / f"distill_strength_1p0__seed_{seed}"
    return directory / "config.yaml", directory / "fingerprint.json", directory / f"epoch_{EPOCHS[seed]:02d}.pt"


def validate() -> dict[str, Any]:
    receipt = _freeze_gate()
    rows = []
    reference_paths: list[str] | None = None
    for seed in SEEDS:
        config_path, fingerprint_path, checkpoint_path = _checkpoint_inputs(seed)
        missing = [str(p) for p in (config_path, fingerprint_path, checkpoint_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing frozen M_T1 inputs: {missing}")
        config = load_config(config_path)
        identity = (
            int(config["data"]["image_size"]),
            config["model"]["vision_encoder"], config["model"]["text_encoder"],
            config["recipe"]["image_token_aggregation"], config["recipe"]["text_token_aggregation"],
            config["data"]["train_captions_per_image"], int(config["training"]["batch_size"]),
        )
        expected = (224, "dinov3_vits16", "all_minilm_l6_v2", "transformer_128", "learned_query_attention", None, 1024)
        if identity != expected:
            raise RuntimeError(f"seed {seed} is not frozen M_T1: {identity}")
        if reference_paths is None:
            reference_paths = [str(config["data"]["train_csv"]), str(config["data"]["image_root"])]
        elif reference_paths != [str(config["data"]["train_csv"]), str(config["data"]["image_root"])]:
            raise RuntimeError("M_T1 seeds do not share the mining pool")
        rows.append({
            "seed": seed, "epoch": EPOCHS[seed],
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "fingerprint_sha256": sha256_file(fingerprint_path),
        })
    result = {
        "status": "READY", "freeze_receipt_sha256": _sha_text(RECEIPT),
        "preregistration_sha256": receipt["files"][str(PREREG.relative_to(ROOT))],
        "candidate_depth": CANDIDATE_DEPTH, "negatives_per_owner": NEGATIVES_PER_OWNER,
        "thresholds": {"image_text": IMAGE_TEXT_THRESHOLD, "text_text": TEXT_TEXT_THRESHOLD},
        "seeds": rows, "flickr_test_used": False,
    }
    atomic_json(result, OUT / "validation.json")
    return result


@torch.inference_mode()
def encode(seed: int) -> dict[str, Any]:
    validation = validate()
    destination = OUT / f"embeddings/m_t1_seed_{seed}.pt"
    metadata_path = OUT / f"embeddings/m_t1_seed_{seed}.json"
    if destination.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        if (
            existing.get("status") in {"COMPLETE", "RESUMED"}
            and int(existing.get("seed", -1)) == seed
            and int(existing.get("epoch", -1)) == EPOCHS[seed]
            and existing.get("artifact_sha256") == sha256_file(destination)
        ):
            return {**existing, "status": "RESUMED"}
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    config_path, fingerprint_path, checkpoint_path = _checkpoint_inputs(seed)
    config = load_config(config_path)
    fingerprint = read_fingerprint(fingerprint_path)
    if fingerprint is None:
        raise RuntimeError(f"invalid fingerprint: {fingerprint_path}")
    model = build_model(config).to(device).eval()
    load_training_checkpoint(checkpoint_path, model, device=device, expected_fingerprint=fingerprint)
    evaluation = deepcopy(config)
    evaluation["data"].update({
        "val_csv": str(config["data"]["train_csv"]), "val_captions_per_image": None,
        "num_workers": int(os.environ.get("SLURM_CPUS_PER_TASK", "12")),
        "persistent_workers": False,
    })
    evaluation["training"]["batch_size"] = ENCODE_BATCH_SIZE
    _, loader = build_dataloaders(evaluation)
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    image_paths: list[str] = []
    text_image_paths: list[str] = []
    captions: list[str] = []
    amp = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with amp:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            batch_captions = [str(v) for v in batch["captions"]]
            outputs = model(images, batch_captions)
            image_embeddings.append(outputs["image_embeds"].detach().cpu().half())
            text_embeddings.append(outputs["text_embeds"].detach().cpu().half())
            image_paths.extend(str(v) for v in batch["image_paths"])
            text_image_paths.extend(str(v) for v in batch["text_image_paths"])
            captions.extend(batch_captions)
    manifest_digest = hashlib.sha256(json.dumps({
        "image_paths": image_paths, "text_image_paths": text_image_paths, "captions": captions,
    }, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "image_embeddings": torch.cat(image_embeddings), "text_embeddings": torch.cat(text_embeddings),
        "image_paths": image_paths, "text_image_paths": text_image_paths, "captions": captions,
        "manifest_digest": manifest_digest,
        "seed": seed, "epoch": EPOCHS[seed], "checkpoint_sha256": sha256_file(checkpoint_path),
        "fingerprint": fingerprint.digest, "validation_sha256": _sha_text(OUT / "validation.json"),
    }
    _atomic_torch(payload, destination)
    result = {
        "status": "COMPLETE", "seed": seed, "epoch": EPOCHS[seed],
        "images": len(image_paths), "captions": len(captions), "manifest_digest": manifest_digest,
        "artifact": str(destination.relative_to(ROOT)), "artifact_sha256": sha256_file(destination),
        "node": provenance["node"], "gpu_model": provenance["gpu_model"], "precision": "native_bf16",
        "flickr_test_used": False,
    }
    atomic_json(result, metadata_path)
    return result


def _caption_owner_arrays(image_paths: list[str], text_paths: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    owner_index = {value: i for i, value in enumerate(image_paths)}
    owners = np.asarray([owner_index[value] for value in text_paths], dtype=np.int64)
    grouped: list[list[int]] = [[] for _ in image_paths]
    for text_index, owner in enumerate(owners.tolist()):
        grouped[owner].append(text_index)
    return owners, [np.asarray(v, dtype=np.int64) for v in grouped]


def _padded_caption_indices(groups: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, groups))
    indices = torch.zeros((len(groups), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(groups), width), dtype=torch.bool, device=device)
    for owner, values in enumerate(groups):
        indices[owner, : len(values)] = torch.as_tensor(values, device=device)
        mask[owner, : len(values)] = True
    return indices, mask


def _scatter_owner_max(scores: torch.Tensor, owners: torch.Tensor, owner_count: int) -> torch.Tensor:
    result = torch.full((scores.shape[0], owner_count), -torch.inf, device=scores.device)
    return result.scatter_reduce(1, owners.unsqueeze(0).expand(scores.shape[0], -1), scores, reduce="amax", include_self=True)


def _guard_block(
    query: torch.Tensor, candidates: torch.Tensor, teacher_images: torch.Tensor,
    teacher_texts: torch.Tensor, caption_indices: torch.Tensor, caption_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Shapes after flattening pairs: query/candidates [P], caption indices [owners,C].
    q_img = teacher_images[query]
    c_img = teacher_images[candidates]
    q_idx, c_idx = caption_indices[query], caption_indices[candidates]
    q_mask, c_mask = caption_mask[query], caption_mask[candidates]
    q_txt, c_txt = teacher_texts[q_idx], teacher_texts[c_idx]
    i2t = torch.einsum("pd,pcd->pc", q_img, c_txt).masked_fill(~c_mask, -torch.inf).amax(1)
    t2i = torch.einsum("pcd,pd->pc", q_txt, c_img).masked_fill(~q_mask, -torch.inf).amax(1)
    cross = torch.maximum(i2t, t2i)
    text = torch.einsum("pqd,pcd->pqc", q_txt, c_txt)
    valid = q_mask.unsqueeze(2) & c_mask.unsqueeze(1)
    text = text.masked_fill(~valid, -torch.inf).flatten(1).amax(1)
    return cross, text, (cross >= IMAGE_TEXT_THRESHOLD) | (text >= TEXT_TEXT_THRESHOLD)


def _load_embedding_set() -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    payloads = []
    for seed in SEEDS:
        path = OUT / f"embeddings/m_t1_seed_{seed}.pt"
        metadata = OUT / f"embeddings/m_t1_seed_{seed}.json"
        if not path.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"missing seed-{seed} mining embeddings")
        recorded = json.loads(metadata.read_text())
        if sha256_file(path) != recorded["artifact_sha256"]:
            raise RuntimeError(f"seed-{seed} embedding checksum mismatch")
        payloads.append(torch.load(path, map_location="cpu", weights_only=False))
    digests = {p["manifest_digest"] for p in payloads}
    if len(digests) != 1:
        raise RuntimeError("seed embedding manifests differ")
    first = payloads[0]
    return payloads, first["image_paths"], first["text_image_paths"], first["captions"]


@torch.inference_mode()
def mine() -> dict[str, Any]:
    validation = validate()
    provenance = _hardware_guard({"resources": {}})
    device = provenance.pop("device")
    payloads, image_paths, text_paths, captions = _load_embedding_set()
    embedding_hashes = {
        str(seed): sha256_file(OUT / f"embeddings/m_t1_seed_{seed}.pt") for seed in SEEDS
    }
    mining_identity = hashlib.sha256(json.dumps({
        "freeze_receipt": _sha_text(RECEIPT), "embeddings": embedding_hashes,
        "candidate_depth": CANDIDATE_DEPTH, "negatives": NEGATIVES_PER_OWNER,
        "image_text_threshold": IMAGE_TEXT_THRESHOLD, "text_text_threshold": TEXT_TEXT_THRESHOLD,
        "block_size": MINE_BLOCK_SIZE,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Exact duplicate captions owned by the same image are one semantic candidate.
    keep: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, pair in enumerate(zip(text_paths, captions)):
        if pair not in seen:
            seen.add(pair); keep.append(index)
    text_paths = [text_paths[i] for i in keep]
    captions = [captions[i] for i in keep]
    for payload in payloads:
        payload["text_embeddings"] = payload["text_embeddings"][keep]
    owners_np, groups_np = _caption_owner_arrays(image_paths, text_paths)
    owners = torch.as_tensor(owners_np, device=device)
    caption_indices, caption_mask = _padded_caption_indices(groups_np, device)
    seed_images = [torch.nn.functional.normalize(p["image_embeddings"].float(), dim=-1).to(device) for p in payloads]
    seed_texts = [torch.nn.functional.normalize(p["text_embeddings"].float(), dim=-1).to(device) for p in payloads]

    teacher_path = ROOT / "results/alignment_wave1/mobileclip2/teacher_cache/mobileclip2_s0_dfndr2b.pt"
    calibration = json.loads(CALIBRATION.read_text())
    if sha256_file(teacher_path) != calibration["source_cache_sha256"]:
        raise RuntimeError("MobileCLIP teacher cache differs from frozen calibration")
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher_image_index = {str(Path(v).resolve()): i for i, v in enumerate(teacher["image_paths"])}
    teacher_text_index = {str(v): i for i, v in enumerate(teacher["text_keys"])}
    t_image_rows = [teacher_image_index[str(Path(v).resolve())] for v in image_paths]
    t_text_rows = [teacher_text_index[_key(owner, caption)] for owner, caption in zip(text_paths, captions)]
    teacher_images = torch.nn.functional.normalize(teacher["image_embeddings"][t_image_rows].float(), dim=-1).to(device)
    teacher_texts = torch.nn.functional.normalize(teacher["text_embeddings"][t_text_rows].float(), dim=-1).to(device)

    naive_rows: list[dict[str, Any]] = []
    guarded_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    owner_count = len(image_paths)
    for start in range(0, owner_count, MINE_BLOCK_SIZE):
        stop = min(start + MINE_BLOCK_SIZE, owner_count)
        block_path = OUT / f"blocks/block_{start:06d}_{stop:06d}.pt"
        if block_path.is_file():
            saved = torch.load(block_path, map_location="cpu", weights_only=False)
            if saved.get("mining_identity") != mining_identity or saved.get("range") != [start, stop]:
                raise RuntimeError(f"stale or mismatched mining resume block: {block_path}")
            naive_rows.extend(saved["naive_rows"])
            guarded_rows.extend(saved["guarded_rows"])
            diagnostic_rows.extend(saved["diagnostic_rows"])
            print(f"resumed owners {stop}/{owner_count}", flush=True)
            continue
        query = torch.arange(start, stop, device=device)
        i2t_scores = None
        for images, texts in zip(seed_images, seed_texts):
            value = images[query] @ texts.T
            i2t_scores = value if i2t_scores is None else i2t_scores + value
        i2t_owner = _scatter_owner_max(i2t_scores / len(SEEDS), owners, owner_count)
        q_indices = caption_indices[query]
        q_mask = caption_mask[query]
        flat = q_indices[q_mask]
        flat_query = torch.arange(len(query), device=device).unsqueeze(1).expand_as(q_indices)[q_mask]
        t2i_scores = None
        for images, texts in zip(seed_images, seed_texts):
            value = texts[flat] @ images.T
            t2i_scores = value if t2i_scores is None else t2i_scores + value
        directional = t2i_scores / len(SEEDS)
        t2i_owner = torch.full((len(query), owner_count), -torch.inf, device=device)
        t2i_owner = t2i_owner.scatter_reduce(0, flat_query.unsqueeze(1).expand_as(directional), directional, reduce="amax", include_self=True)
        symmetric = 0.5 * (i2t_owner + t2i_owner)
        symmetric[torch.arange(len(query), device=device), query] = -torch.inf
        scores, candidates = symmetric.topk(CANDIDATE_DEPTH, dim=1)
        pair_query = query.unsqueeze(1).expand_as(candidates).reshape(-1)
        pair_candidates = candidates.reshape(-1)
        cross, text, excluded = _guard_block(pair_query, pair_candidates, teacher_images, teacher_texts, caption_indices, caption_mask)
        cross = cross.view(len(query), CANDIDATE_DEPTH)
        text = text.view(len(query), CANDIDATE_DEPTH)
        excluded = excluded.view(len(query), CANDIDATE_DEPTH)
        block_naive: list[dict[str, Any]] = []
        block_guarded: list[dict[str, Any]] = []
        block_diagnostics: list[dict[str, Any]] = []
        for local, owner in enumerate(range(start, stop)):
            candidate_values = candidates[local].tolist()
            score_values = scores[local].tolist()
            for rank in range(NEGATIVES_PER_OWNER):
                target = int(candidate_values[rank])
                block_naive.append({"query_owner_index": owner, "negative_owner_index": target, "rank": rank + 1, "score": float(score_values[rank])})
            valid_positions = (~excluded[local]).nonzero(as_tuple=False).flatten().tolist()
            for retained_rank, position in enumerate(valid_positions[:NEGATIVES_PER_OWNER], 1):
                target = int(candidate_values[position])
                block_guarded.append({"query_owner_index": owner, "negative_owner_index": target, "rank": retained_rank, "source_rank": position + 1, "score": float(score_values[position]), "teacher_cross_modal_similarity": float(cross[local, position]), "teacher_caption_similarity": float(text[local, position])})
            block_diagnostics.append({
                "query_owner_index": owner, "query_image_path": image_paths[owner],
                "top64": CANDIDATE_DEPTH, "cross_modal_excluded": int((cross[local] >= IMAGE_TEXT_THRESHOLD).sum()),
                "caption_excluded": int((text[local] >= TEXT_TEXT_THRESHOLD).sum()),
                "union_excluded": int(excluded[local].sum()), "retained": int((~excluded[local]).sum()),
                "has_required_eight": len(valid_positions) >= NEGATIVES_PER_OWNER,
            })
        _atomic_torch({
            "mining_identity": mining_identity, "range": [start, stop],
            "naive_rows": block_naive, "guarded_rows": block_guarded,
            "diagnostic_rows": block_diagnostics,
        }, block_path)
        naive_rows.extend(block_naive)
        guarded_rows.extend(block_guarded)
        diagnostic_rows.extend(block_diagnostics)
        print(f"mined owners {stop}/{owner_count}", flush=True)

    diagnostics = pd.DataFrame(diagnostic_rows)
    feasible = bool(diagnostics["has_required_eight"].all())
    atomic_csv(pd.DataFrame(naive_rows), OUT / "naive_graph.csv")
    atomic_csv(pd.DataFrame(guarded_rows), OUT / "guarded_graph.csv")
    atomic_csv(diagnostics, OUT / "exclusion_diagnostics.csv")
    summary = {
        "status": "FEASIBLE" if feasible else "INFEASIBLE_UNDER_PREREGISTERED_GUARD",
        "training_authorized": feasible, "owners": owner_count,
        "unique_captions": len(captions), "duplicate_caption_rows_removed": len(payloads[0]["captions"]) - len(captions),
        "candidate_depth": CANDIDATE_DEPTH, "negatives_per_owner": NEGATIVES_PER_OWNER,
        "queries_with_eight": int(diagnostics["has_required_eight"].sum()),
        "queries_without_eight": int((~diagnostics["has_required_eight"]).sum()),
        "exclusion": {
            "cross_modal_total": int(diagnostics["cross_modal_excluded"].sum()),
            "caption_total": int(diagnostics["caption_excluded"].sum()),
            "union_total": int(diagnostics["union_excluded"].sum()),
            "union_rate": float(diagnostics["union_excluded"].sum() / (owner_count * CANDIDATE_DEPTH)),
        },
        "artifacts": {
            "naive_graph": "results/guarded_hard_negative_study/mining/naive_graph.csv",
            "guarded_graph": "results/guarded_hard_negative_study/mining/guarded_graph.csv",
            "diagnostics": "results/guarded_hard_negative_study/mining/exclusion_diagnostics.csv",
        },
        "validation_sha256": _sha_text(OUT / "validation.json"),
        "freeze_receipt_sha256": _sha_text(RECEIPT),
        "mining_identity": mining_identity, "embedding_sha256": embedding_hashes,
        "node": provenance["node"], "gpu_model": provenance["gpu_model"], "precision": "native_bf16",
        "flickr_test_used": False,
    }
    for name, relative in summary["artifacts"].items():
        summary["artifacts"][name] = {"path": relative, "sha256": sha256_file(ROOT / relative)}
    atomic_json(summary, OUT / "report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "encode", "mine"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate()
    elif args.command == "encode":
        if args.seed is None:
            parser.error("encode requires --seed")
        result = encode(args.seed)
    else:
        result = mine()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
