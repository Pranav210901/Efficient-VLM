from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.alignment_v3.fingerprint import sha256_file
from src.alignment_v3.probe1_error_decomposition import (
    _assert_blinded_export,
    _bin,
    _intersection,
    _metric_payload,
    _stable_order,
    _test_seal_guard,
    rank_payload,
    verify_freeze,
)


def _gallery() -> dict:
    images = [
        {"gallery_index": 0, "image_id": "a", "image_path": "a.jpg"},
        {"gallery_index": 1, "image_id": "b", "image_path": "b.jpg"},
    ]
    captions = []
    for image_id in ("a", "b"):
        for index in range(5):
            captions.append(
                {
                    "caption_id": f"{image_id}::caption_{index}",
                    "image_id": image_id,
                    "caption": f"{image_id} caption {index}",
                }
            )
    return {"images": images, "captions": captions}


def test_rank_payload_retains_all_five_positives_and_uses_best_rank() -> None:
    gallery = _gallery()
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor([[0.7, 0.3]] * 5 + [[0.3, 0.7]] * 5)
    payload = {
        "image_embeds": image,
        "text_embeds": text,
        "image_paths": ["a.jpg", "b.jpg"],
        "text_image_paths": ["a.jpg"] * 5 + ["b.jpg"] * 5,
    }
    frame = rank_payload(payload, gallery, "toy", 42)
    i2t = frame[frame.direction.eq("i2t")]
    assert len(i2t) == 2
    for row in i2t.itertuples():
        ranks = json.loads(row.all_positive_ranks)
        positives = json.loads(row.all_positive_ids)
        assert len(ranks) == len(positives) == 5
        assert row.positive_rank == min(ranks)
    assert len(frame[frame.direction.eq("t2i")]) == 10


def test_exact_ties_resolve_by_frozen_target_index() -> None:
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1]])
    assert _stable_order(scores).tolist() == [[0, 1, 2, 3]]


def test_metric_partition_is_exhaustive() -> None:
    frame = pd.DataFrame(
        {
            "positive_rank": [1, 2, 10, 11],
            "partition": ["rank_1", "rank_2_10", "rank_2_10", "outside_top_10"],
        }
    )
    result = _metric_payload(frame)
    assert result["R@1"] == 0.25
    assert result["R@10"] == 0.75
    assert sum(value["count"] for value in result["failure_partition"].values()) == 4


def test_intersection_sets_are_exhaustive() -> None:
    c4 = pd.DataFrame({"query_id": list("abcd"), "positive_rank": [1, 5, 12, 20]})
    control = pd.DataFrame({"query_id": list("abcd"), "positive_rank": [2, 11, 3, 20]})
    result = _intersection(c4, control)
    assert result["intersection_size"] == 1
    assert result["c4_exclusive_size"] == 1
    assert result["openclip_exclusive_size"] == 1
    assert result["neither_size"] == 1


def test_bin_boundaries_are_frozen_left_closed() -> None:
    edges = [0.2, 0.7]
    assert _bin(0.2, edges) == "low"
    assert _bin(0.7, edges) == "medium"
    assert _bin(0.70001, edges) == "high"


def test_freeze_receipt_aborts_after_ambiguity_mutation(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery_manifest.json"
    ambiguity = tmp_path / "ambiguity_bins.json"
    gallery.write_text("{}\n")
    ambiguity.write_text("{}\n")
    receipt = {
        "gallery_manifest_sha256": sha256_file(gallery),
        "ambiguity_bins_sha256": sha256_file(ambiguity),
        "retrieval_artifacts_present_at_freeze": False,
    }
    (tmp_path / "ambiguity_freeze_receipt.json").write_text(json.dumps(receipt))
    pipeline = {"output_root": str(tmp_path)}
    assert verify_freeze(pipeline)["status"] == "PASS"
    ambiguity.write_text('{"changed": true}\n')
    with pytest.raises(RuntimeError, match="changed after freeze"):
        verify_freeze(pipeline)


def test_flickr_test_path_is_rejected() -> None:
    pipeline = {
        "gallery": {
            "csv": "data/flickr30k/test.csv",
            "sealed_test_csv": "data/flickr30k/test.csv",
            "sealed_test_dataset_id": "flickr30k_karpathy_test",
        }
    }
    with pytest.raises(RuntimeError, match="sealed"):
        _test_seal_guard(pipeline)


def test_blinded_export_rejects_answer_leakage() -> None:
    _assert_blinded_export(pd.DataFrame([{"audit_id": "a", "query_caption": "caption"}]))
    with pytest.raises(RuntimeError, match="leaks answer fields"):
        _assert_blinded_export(pd.DataFrame([{"audit_id": "a", "positive_rank": 1}]))
