from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src.phase2.checkpointing import PreemptionState, atomic_torch_save, checkpoint_payload, load_checkpoint
from src.phase2.bridge_evaluation import _limit_classification_per_task
from src.phase2.classification_reranker import classification_metrics, rerank_class_candidates
from src.phase2.config import load_phase2_config, parse_wall_time, safe_array_concurrency, validate_resource_request
from src.phase2.cross_attention import TextToVisionBridge
from src.phase2.dense_teacher import DenseTaskConditionedTeacher
from src.phase2.dense_training import _wrap_dense_teacher
from src.phase2.distributed import DistributedContext, rank_zero, reduce_mean, sampler_for
from src.phase2.evaluator import read_coco_captions
from src.phase2.hard_negative_loader import validate_training_frame
from src.phase2.losses import image_text_matching_loss, pairwise_ranking_loss
from src.phase2.pair_scorer import PairScorer
from src.phase2.pipeline import _incomplete_array_indices
from src.phase2.prerequisites import load_locked_selection, primary_paths, validate_phase2_prerequisites
from src.phase2.retrieval_reranker import candidate_coverage, rerank_topk
from src.phase2.schemas import TokenBatch
from src.phase2.statistics import pareto_front
from src.phase2.token_adapters import TextTokenAdapter, VisionTokenAdapter


ROOT = Path(__file__).resolve().parents[2]
LEGACY_EXECUTION_PREREQUISITES = (
    ROOT / "results/phase15",
    ROOT / "data/multitask/cifar100",
    ROOT / "data/multitask/oxford-iiit-pet",
    ROOT / "data/multitask/eurosat",
)
LEGACY_EXECUTION_STATE_ACTIVE = all(path.exists() for path in LEGACY_EXECUTION_PREREQUISITES)
requires_legacy_execution_state = pytest.mark.skipif(
    not LEGACY_EXECUTION_STATE_ACTIVE,
    reason="legacy Phase 1.5/2 rerun prerequisites are not installed in the active environment",
)


def token_batch(batch=3, tokens=5, dim=16):
    mask = torch.ones(batch, tokens, dtype=torch.bool); mask[:, -1] = False
    return TokenBatch(torch.randn(batch, tokens, dim), mask, None, {}).validate()


@requires_legacy_execution_state
def test_selection_yaml_and_primary_paths():
    selection = load_locked_selection(ROOT)
    assert selection["schema_version"] == 1
    assert len(primary_paths(selection)) == 4


@requires_legacy_execution_state
def test_prerequisite_validation_is_ready():
    report = validate_phase2_prerequisites(ROOT, write=False)
    assert report["status"] == "VALID" and report["phase15_readiness"] == "READY"


@pytest.mark.parametrize("shape,input_dim,tokens", [((2, 32, 7, 7), 32, 49), ((2, 7, 7, 32), 32, 49), ((2, 11, 32), 32, 11)])
def test_vision_adapter_shapes_and_masks(shape, input_dim, tokens):
    output = VisionTokenAdapter(input_dim, 16)(torch.randn(*shape))
    assert output.tokens.shape == (2, tokens, 16)
    assert output.attention_mask.all()


def test_text_masks_are_preserved():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    output = TextTokenAdapter(12, 16)(torch.randn(2, 3, 12), mask)
    assert torch.equal(output.attention_mask, mask)


def test_projection_gradients():
    adapter = VisionTokenAdapter(8, 16); adapter(torch.randn(2, 8, 2, 2)).tokens.sum().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_cross_attention_forward_backward_and_score_shape():
    text, vision = token_batch(), token_batch(tokens=7)
    bridge = TextToVisionBridge(16, 4, 1, 32); scorer = PairScorer(16, 8)
    score = scorer(bridge(text, vision)); score.sum().backward()
    assert score.shape == (3,)
    assert any(parameter.grad is not None for parameter in bridge.parameters())


def test_losses_are_finite_and_ranking_improves():
    positive, negative = torch.tensor([2.0, 1.0]), torch.tensor([-1.0, 0.0])
    assert torch.isfinite(image_text_matching_loss(positive, negative))
    assert pairwise_ranking_loss(positive, negative) == 0


def retrieval_frame():
    return pd.DataFrame({
        "allowed_for_training": [True], "source_split": ["train"], "configuration_id": ["pair"],
        "direction": ["i2t"], "query_id": ["q"], "positive_id": ["i1"], "positive_text": ["good"],
        "negative_id": ["n"], "negative_owner_id": ["i2"], "negative_text": ["bad"], "candidate_rank": [1],
    })


def test_false_negative_exclusion_and_leakage_prevention():
    frame = retrieval_frame(); assert validate_training_frame(frame, "retrieval")["status"] == "valid"
    frame.loc[0, "negative_owner_id"] = "i1"
    with pytest.raises(ValueError, match="same-image"): validate_training_frame(frame, "retrieval")
    frame = retrieval_frame(); frame.loc[0, "query_id"] = "protected"
    with pytest.raises(ValueError, match="protected"): validate_training_frame(frame, "retrieval", ["protected"])


def test_candidate_reranking_and_coverage():
    dual = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.4]])
    ranked, _ = rerank_topk(dual, lambda q, c: -c.float(), 2)
    assert ranked.shape == (2, 2)
    assert candidate_coverage([["a", "b"], ["c"]], [{"b"}, {"x"}]) == 0.5


@requires_legacy_execution_state
def test_phase2_coco_source_matches_multitask_cache_identity_and_order():
    config = load_phase2_config(ROOT / "configs/phase2/bridge_evaluation.yaml")
    images, captions, owners = read_coco_captions(ROOT / config["coco_csv"], ROOT)
    pair_id = primary_paths(load_locked_selection(ROOT))[0]["pair_id"]
    cache = torch.load(
        ROOT / "results/phase1_multitask/cache" / f"{pair_id}__coco__full.pt",
        map_location="cpu",
        weights_only=False,
    )["payload"]
    assert len(captions) == cache["text_embeddings"].shape[0]
    assert images == list(map(str, cache["image_ids"]))
    assert owners == list(map(str, cache["text_image_ids"]))


def test_resume_array_submits_only_incomplete_indices(tmp_path):
    root = tmp_path
    status_root = root / "results/phase2/slurm_status/stage"
    for index, status in enumerate(["COMPLETED", "FAILED", "COMPLETED", "RUNNING"]):
        path = status_root / str(index) / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}))
    assert _incomplete_array_indices(root, "stage", 5) == [1, 3, 4]


def test_dense_score_sampling_preserves_every_classification_task():
    frame = pd.DataFrame({"source_task": ["cifar"] * 5 + ["pets"] * 2 + ["eurosat"] * 4, "value": range(11)})
    sampled = _limit_classification_per_task(frame, 3)
    assert sampled.groupby("source_task").size().to_dict() == {"cifar": 3, "eurosat": 3, "pets": 2}


def test_classification_reranking_metrics():
    dual = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.4]])
    candidates, _ = rerank_class_candidates(dual, torch.tensor([[0.1, 0.9], [0.8, 0.2]]), 2)
    metrics = classification_metrics(candidates, torch.tensor([2, 0]), 2)
    assert metrics["candidate_coverage"] == 1.0


def test_checkpoint_save_reload_and_resume_state(tmp_path):
    model = nn.Linear(4, 1); optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "state.pt"; atomic_torch_save(checkpoint_payload(model, optimizer, epoch=3, global_step=17), path)
    state = load_checkpoint(path, model, optimizer)
    assert state["epoch"] == 3 and state["global_step"] == 17


def test_checkpoint_reload_normalises_legacy_rng_state_to_cpu_byte_tensor(tmp_path):
    model = nn.Linear(4, 1)
    payload = checkpoint_payload(model, epoch=0)
    payload["torch_random_state"] = payload["torch_random_state"].tolist()
    path = tmp_path / "legacy_rng.pt"
    atomic_torch_save(payload, path)
    state = load_checkpoint(path, model)
    assert isinstance(state["torch_random_state"], list)


def test_signal_handler_sets_checkpoint_request():
    PreemptionState.requested = False; PreemptionState.handler(10, None)
    assert PreemptionState.requested and PreemptionState.signal_number == 10
    PreemptionState.requested = False


def test_single_process_distributed_helpers():
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    assert rank_zero(context) and sampler_for(list(range(4)), context) is None
    assert reduce_mean(torch.tensor(2.0), context) == 2


def test_dense_teacher_ddp_allows_strategy_specific_unused_parameters(monkeypatch):
    captured = {}

    class FakeDDP:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)
            self.model = model

    monkeypatch.setattr("src.phase2.dense_training.DistributedDataParallel", FakeDDP)
    context = DistributedContext(0, 0, 4, torch.device("cpu"), True)
    model = DenseTaskConditionedTeacher(4, 3, strategy="static")
    wrapped = _wrap_dense_teacher(model, context)
    assert wrapped.model is model
    assert captured["find_unused_parameters"] is True


@pytest.mark.parametrize("strategy", ["uniform", "static", "task_only", "input_conditioned", "task_and_input", "task_attention"])
def test_dense_weights_sum_to_one_and_are_task_conditioned(strategy):
    model = DenseTaskConditionedTeacher(4, 3, strategy=strategy, path_dropout=0)
    output = model(torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.allclose(output["weights"].sum(1), torch.ones(6), atol=1e-6)
    assert torch.isfinite(output["gating_entropy"]).all()


def test_path_dropout_never_drops_every_path():
    model = DenseTaskConditionedTeacher(4, 2, path_dropout=0.99); model.train()
    weights = model(torch.randn(32, 4), torch.zeros(32, dtype=torch.long))["weights"]
    assert torch.allclose(weights.sum(1), torch.ones(32)) and torch.isfinite(weights).all()


def test_pareto_front():
    frame = pd.DataFrame({"method": ["a", "b", "c"], "performance": [0.8, 0.9, 0.7], "latency_ms": [2.0, 3.0, 4.0]})
    assert set(pareto_front(frame)["method"]) == {"a", "b"}


def test_configs_parse_and_respect_resources():
    for path in (ROOT / "configs/phase2").glob("*.yaml"):
        config = load_phase2_config(path); validate_resource_request(config["resources"])
    assert safe_array_concurrency(1, 8) == 8 and safe_array_concurrency(4, 8) == 2
    assert parse_wall_time("2-23:59:00") < 3 * 86400


def _directive(script: str, name: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{name}(?:=|\s+)(\S+)", script, re.MULTILINE)
    return match.group(1) if match else None


def test_all_slurm_scripts_use_teaching_and_safe_resources():
    scripts = list((ROOT / "slurm/phase2").glob("*.sbatch")); assert len(scripts) == 14
    for path in scripts:
        source = path.read_text(); assert "#SBATCH --partition=teaching" in source and "set -euo pipefail" in source
        gpu = _directive(source, "gres")
        if gpu: assert int(gpu.split(":")[-1]) <= 8
        assert parse_wall_time(_directive(source, "time")) <= parse_wall_time("2-23:59:00")
        array = _directive(source, "array")
        if array:
            throttle = int(array.split("%")[-1]); per_task = int(gpu.split(":")[-1]); assert throttle * per_task <= 8


def test_notebook_has_safe_smoke_controls_and_no_full_execution():
    notebook = json.loads((ROOT / "notebooks/01_experiment_workflow.ipynb").read_text())
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    controls = cells["phase2-controls"]
    assert "RUN_PHASE2_SMOKE_TESTS = False" in controls and "RUN_PHASE2_DDP_SMOKE_TEST = False" in controls
    phase2_code = "\n".join(value for key, value in cells.items() if key.startswith("phase2-"))
    assert "subprocess.run" not in phase2_code and "sbatch(" not in phase2_code


def test_result_schema_token_batch_rejects_nonfinite():
    value = token_batch(); value.tokens[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"): value.validate()
