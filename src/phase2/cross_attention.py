from __future__ import annotations

import torch
from torch import nn

from .schemas import TokenBatch
from .token_adapters import masked_mean


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, feedforward_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("bridge dimension must be divisible by attention heads")
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.post_attention = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, feedforward_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, text: TokenBatch, vision: TokenBatch) -> torch.Tensor:
        query = self.query_norm(text.tokens)
        memory = self.memory_norm(vision.tokens)
        attended, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=~vision.attention_mask,
            need_weights=False,
        )
        value = self.post_attention(text.tokens + attended)
        return self.output_norm(value + self.feedforward(value))


class LearnedPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bnd,d->bn", tokens, self.query)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.einsum("bn,bnd->bd", logits.softmax(-1), tokens)


class TextToVisionBridge(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        layers: int = 1,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "masked_mean",
    ) -> None:
        super().__init__()
        if layers not in {1, 2}:
            raise ValueError("Phase 2 bridge supports one or two lightweight layers")
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(dim, heads, feedforward_dim, dropout) for _ in range(layers)]
        )
        self.pooling = pooling
        self.learned_pool = LearnedPool(dim) if pooling == "learned" else None

    def forward(self, text: TokenBatch, vision: TokenBatch) -> torch.Tensor:
        current = text
        for layer in self.layers:
            values = layer(current, vision)
            current = TokenBatch(values, text.attention_mask, metadata=text.metadata)
        if self.pooling == "masked_mean":
            return masked_mean(current.tokens, current.attention_mask)
        if self.pooling == "cls":
            return current.tokens[:, 0]
        if self.pooling == "learned" and self.learned_pool is not None:
            return self.learned_pool(current.tokens, current.attention_mask)
        raise ValueError(f"Unknown pooling strategy {self.pooling!r}")
