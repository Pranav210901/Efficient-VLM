from __future__ import annotations

import torch
from torch import nn


class DenseTaskConditionedTeacher(nn.Module):
    """Combine scores from all selected bridge paths with normalised weights."""

    def __init__(
        self,
        paths: int,
        tasks: int,
        strategy: str = "task_and_input",
        hidden_dim: int = 64,
        temperature: float = 1.0,
        path_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if paths < 1 or tasks < 1:
            raise ValueError("paths and tasks must be positive")
        self.paths, self.strategy = int(paths), strategy
        self.temperature = float(temperature)
        self.path_dropout = float(path_dropout)
        self.static_logits = nn.Parameter(torch.zeros(paths))
        self.task_embedding = nn.Embedding(tasks, hidden_dim)
        self.task_gate = nn.Linear(hidden_dim, paths)
        self.input_gate = nn.Sequential(nn.Linear(paths, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, paths))
        self.joint_gate = nn.Sequential(nn.Linear(hidden_dim + paths, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, paths))
        self.concat_head = nn.Sequential(nn.Linear(paths, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.task_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.path_keys = nn.Parameter(torch.randn(paths, hidden_dim) * 0.02)

    def path_weights(self, path_scores: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        task = self.task_embedding(task_ids)
        if self.strategy in {"uniform", "best_single", "concat_mlp"}:
            logits = torch.zeros_like(path_scores)
            if self.strategy == "best_single": logits[:, 0] = 20.0
        elif self.strategy == "static":
            logits = self.static_logits.expand_as(path_scores)
        elif self.strategy == "task_only":
            logits = self.task_gate(task)
        elif self.strategy == "input_conditioned":
            logits = self.input_gate(path_scores)
        elif self.strategy == "task_and_input":
            logits = self.joint_gate(torch.cat((path_scores, task), dim=-1))
        elif self.strategy == "task_attention":
            query = task.unsqueeze(1)
            memory = self.path_keys.unsqueeze(0).expand(path_scores.shape[0], -1, -1)
            attended, weights = self.task_attention(query, memory, memory, need_weights=True)
            del attended
            logits = weights.squeeze(1).clamp_min(1e-8).log() + path_scores.detach()
        else:
            raise ValueError(f"Unknown dense strategy {self.strategy!r}")
        if self.training and self.path_dropout > 0 and self.paths > 1:
            dropped = torch.rand_like(logits).lt(self.path_dropout)
            all_dropped = dropped.all(dim=1)
            dropped[all_dropped, 0] = False
            logits = logits.masked_fill(dropped, torch.finfo(logits.dtype).min)
        return (logits / max(self.temperature, 1e-6)).softmax(dim=-1)

    def forward(self, path_scores: torch.Tensor, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if path_scores.ndim != 2 or path_scores.shape[1] != self.paths:
            raise ValueError(f"path_scores must be [batch,{self.paths}]")
        weights = self.path_weights(path_scores, task_ids)
        score = self.concat_head(path_scores).squeeze(-1) if self.strategy == "concat_mlp" else (weights * path_scores).sum(-1)
        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(-1)
        return {"score": score, "weights": weights, "gating_entropy": entropy, "dominant_path": weights.argmax(-1)}


def teacher_regularisation(weights: torch.Tensor, entropy_weight: float = 0.001, balance_weight: float = 0.01) -> torch.Tensor:
    entropy = -(weights.clamp_min(1e-8).log() * weights).sum(-1).mean()
    balance = (weights.mean(0) - 1.0 / weights.shape[1]).pow(2).mean()
    return balance_weight * balance - entropy_weight * entropy
