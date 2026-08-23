from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


TEXT_MODEL_REGISTRY = {
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert": "distilbert-base-uncased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge_small_en": "BAAI/bge-small-en-v1.5",
    "e5_small_v2": "intfloat/e5-small-v2",
}

TEXT_PREFIX_DEFAULTS = {
    "e5_small_v2": "passage: ",
}


@dataclass(frozen=True)
class TextEncoderInfo:
    name: str
    output_dim: int


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype, device=last_hidden_state.device)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class TextEncoder(nn.Module):
    # pretrained text encoder wrapper

    def __init__(self, name: str, pretrained: bool = True, freeze: bool = True, text_prefix: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.freeze = freeze
        self.text_prefix = TEXT_PREFIX_DEFAULTS.get(name, "") if text_prefix is None else text_prefix
        if name == "openclip_text":
            self.encoder, self.tokenizer, self.output_dim = self._build_openclip(pretrained=pretrained)
            self.is_openclip = True
        else:
            self.encoder, self.tokenizer, self.output_dim = self._build_hf(name=name, pretrained=pretrained)
            self.is_openclip = False
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, captions: list[str]) -> torch.Tensor:
        return self.forward_tokens(self.tokenize(captions))

    @property
    def native_context_length(self) -> int:
        if self.is_openclip:
            value = getattr(self.tokenizer, "context_length", None)
            if value is None:
                value = getattr(getattr(self.encoder, "clip_model", None), "context_length", 77)
            return int(value)
        value = getattr(self.encoder.config, "max_position_embeddings", None)
        if value is None:
            value = getattr(self.tokenizer, "model_max_length", None)
        if value is None or int(value) > 1_000_000:
            raise ValueError(f"cannot determine native context length for {self.name}")
        return int(value)

    def tokenize(
        self,
        captions: list[str],
        *,
        pad_to_native_context: bool = False,
    ) -> object:
        captions = self._prepare_captions(captions)
        if self.is_openclip:
            return self.tokenizer(captions)
        options = {
            "padding": "max_length" if pad_to_native_context else True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if pad_to_native_context:
            options["max_length"] = self.native_context_length
        return self.tokenizer(captions, **options)

    def forward_tokens(self, tokens: object) -> torch.Tensor:
        hidden, attention_mask = self.forward_token_features(tokens)
        return mean_pool(hidden, attention_mask)

    def forward_token_features(
        self, tokens: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # return final non-pooled token states and their validity mask
        tokens = tokens.to(self.device)
        if self.is_openclip:
            raise RuntimeError("OpenCLIP text wrapper does not expose token states")
        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(**tokens)
        else:
            outputs = self.encoder(**tokens)
        return outputs.last_hidden_state, tokens["attention_mask"]

    def _prepare_captions(self, captions: list[str]) -> list[str]:
        if not self.text_prefix:
            return captions
        return [caption if caption.startswith(self.text_prefix) else f"{self.text_prefix}{caption}" for caption in captions]

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _build_hf(name: str, pretrained: bool) -> tuple[nn.Module, object, int]:
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("transformers is required for Hugging Face text encoders.") from exc
        model_name = TEXT_MODEL_REGISTRY.get(name, name)
        model = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return model, tokenizer, int(model.config.hidden_size)

    @staticmethod
    def _build_openclip(pretrained: bool) -> tuple[nn.Module, object, int]:
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("open_clip_torch is required for openclip_text.") from exc
        weights = "openai" if pretrained else None
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=weights)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        class OpenCLIPTextEncoder(nn.Module):
            def __init__(self, clip_model: nn.Module) -> None:
                super().__init__()
                self.clip_model = clip_model

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return self.clip_model.encode_text(tokens)

        output_dim = int(getattr(model, "text_projection", torch.empty(512, 512)).shape[-1])
        return OpenCLIPTextEncoder(model), tokenizer, output_dim
