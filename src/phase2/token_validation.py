from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.transforms import build_image_transform
from src.phase15.io_utils import atomic_csv, atomic_json
from .bridge_model import FrozenPairBridge
from .prerequisites import load_locked_selection, primary_paths


def validate_token_interfaces(project_root: str | Path, config: dict, device: str | None = None, limit: int | None = None) -> dict:
    root = Path(project_root).resolve()
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    paths = primary_paths(load_locked_selection(root))[:limit]
    if not paths: raise RuntimeError("No primary paths were selected")
    retrieval = root / "results/phase15/training_hard_negatives/retrieval_train_hard_negatives.parquet"
    sample = pd.read_parquet(retrieval, columns=["positive_id", "positive_text"]).head(1).iloc[0]
    image_size = int(config.get("image_size", 224)); bridge_dim = int(config.get("bridge_dim", 256))
    image = build_image_transform(image_size, train=False)(Image.open(sample.positive_id).convert("RGB")).unsqueeze(0).to(target)
    rows, memory = [], []
    for pair in paths:
        if target.type == "cuda": torch.cuda.reset_peak_memory_stats(target)
        model = FrozenPairBridge.from_selected_pair(pair, target, bridge_dim=bridge_dim, attention_heads=int(config.get("attention_heads", 4)))
        vision = model.tokens.vision_tokens(image); text = model.tokens.text_tokens([str(sample.positive_text)])
        score = model.forward_tokens(vision, text)["scores"]
        score.sum().backward(); model.validate_gradients()
        rows.append({
            "pair_id": pair["pair_id"], "vision_tokens": vision.tokens.shape[1], "text_tokens": text.tokens.shape[1],
            "bridge_dim": vision.tokens.shape[2], "vision_mask_valid": int(vision.attention_mask.sum()),
            "text_mask_valid": int(text.attention_mask.sum()), "dtype": str(vision.tokens.dtype), "device": str(vision.tokens.device),
            "finite": bool(torch.isfinite(vision.tokens).all() and torch.isfinite(text.tokens).all() and torch.isfinite(score).all()),
            "backbones_frozen": not any(p.requires_grad for p in model.tokens.alignment_model.parameters()), "adapter_gradients": True,
        })
        memory.append({"pair_id": pair["pair_id"], "peak_memory_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else 0})
        del model
    if not all(row["finite"] and row["backbones_frozen"] for row in rows): raise RuntimeError("Token-interface validation failed")
    output = root / "results/phase2/token_interfaces"; output.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(rows), output / "token_interface_manifest.csv")
    atomic_json({"status": "PASS", "interfaces": rows}, output / "token_interface_manifest.json")
    atomic_csv(pd.DataFrame(memory), output / "memory_profile.csv")
    return {"status": "PASS", "pairs": len(rows), "device": str(target)}
