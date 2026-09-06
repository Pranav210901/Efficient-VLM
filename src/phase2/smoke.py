from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

from src.phase15.io_utils import atomic_json
from .checkpointing import atomic_torch_save, checkpoint_payload, load_checkpoint
from .cross_attention import TextToVisionBridge
from .dense_teacher import DenseTaskConditionedTeacher
from .losses import bridge_loss
from .pair_scorer import PairScorer
from .retrieval_reranker import rerank_topk
from .schemas import TokenBatch
from .token_adapters import TextTokenAdapter, VisionTokenAdapter
from .distributed import cleanup_distributed, initialise_distributed, reduce_mean, seed_rank


def run_synthetic_smoke(project_root: str | Path, device: str | None = None) -> dict:
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    vision_adapter = VisionTokenAdapter(32, 64).to(target)
    text_adapter = TextTokenAdapter(24, 64).to(target)
    bridge = TextToVisionBridge(64, 4, 1, 128, 0.1, "masked_mean").to(target)
    scorer = PairScorer(64, 64).to(target)
    vision = vision_adapter(torch.randn(4, 32, 7, 7, device=target))
    mask = torch.tensor([[1,1,1,0], [1,1,0,0], [1,1,1,1], [1,0,0,0]], dtype=torch.bool, device=target)
    text = text_adapter(torch.randn(4, 4, 24, device=target), mask)
    positive_features = bridge(text, vision); positive = scorer(positive_features)
    negative_features = bridge(TokenBatch(text.tokens.flip(0), text.attention_mask.flip(0), None, {}), vision); negative = scorer(negative_features)
    losses = bridge_loss(positive, negative, positive_features, negative_features, contrastive_weight=0.1)
    losses["loss"].backward()
    gradients = sum(parameter.grad is not None for module in (vision_adapter, text_adapter, bridge, scorer) for parameter in module.parameters())
    dense = DenseTaskConditionedTeacher(4, 3).to(target)
    dense_output = dense(torch.randn(5, 4, device=target), torch.tensor([0,1,2,0,1], device=target))
    dual = torch.randn(3, 8, device=target)
    ranked, _ = rerank_topk(dual, lambda q, c: dual[q, c] + c.float() * 0.01, 4)
    combined = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 1))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        atomic_torch_save(checkpoint_payload(combined, epoch=0, global_step=1), path)
        load_checkpoint(path, combined)
        checkpoint_ok = path.exists()
    result = {
        "status": "PASS", "device": str(target), "loss": float(losses["loss"].detach()),
        "trainable_gradients": gradients, "dense_weights_sum_to_one": bool(torch.allclose(dense_output["weights"].sum(-1), torch.ones(5, device=target))),
        "reranked_shape": list(ranked.shape), "checkpoint_roundtrip": checkpoint_ok,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else 0,
    }
    output = Path(project_root) / "results/phase2/smoke"
    output.mkdir(parents=True, exist_ok=True); atomic_json(result, output / "synthetic_smoke.json")
    return result


def run_ddp_smoke(project_root: str | Path) -> dict:
    context = initialise_distributed(); seed_rank(42, context.rank)
    model = nn.Linear(8, 1).to(context.device)
    if context.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[context.local_rank] if context.device.type == "cuda" else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = model(torch.randn(4, 8, device=context.device)).pow(2).mean(); loss.backward(); optimizer.step()
    mean = float(reduce_mean(loss.detach(), context)); result = {"status": "PASS", "world_size": context.world_size, "mean_loss": mean}
    if context.rank == 0:
        output = Path(project_root) / "results/phase2/smoke"; output.mkdir(parents=True, exist_ok=True); atomic_json(result, output / "ddp_smoke.json")
    cleanup_distributed(); return result
