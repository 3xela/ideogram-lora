"""Minimal, dependency-light LoRA for the Ideogram 4 transformer.

Why not PEFT? The released Ideogram 4 model is a bespoke ``nn.Module``
(`ideogram4.modeling_ideogram4.Ideogram4Transformer`), not a diffusers/PEFT-
registered model, and its linears get swapped to bitsandbytes ``Linear4bit``
(nf4) at load time. A tiny manual wrapper attaches cleanly on top of *any* base
layer (nf4 ``Linear4bit``, weight-only ``Fp8Linear``, or plain ``nn.Linear``)
and keeps the whole thing easy to read.

A LoRA layer computes ``base(x) + (dropout(x) @ A^T) @ B^T * (alpha / r)``.
``A`` is initialised with a small normal, ``B`` with zeros, so the adapter is a
no-op at step 0 and the base model is untouched until training moves it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


# Real module families in Ideogram4Transformer.layers[i] (verified against
# ideogram4.modeling_ideogram4). Note the *fused* qkv and the o projection.
TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "attention": ("attention.qkv", "attention.o"),
    "attention_mlp": (
        "attention.qkv",
        "attention.o",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ),
    "all_linear": (
        "attention.qkv",
        "attention.o",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ),
}


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_preset: str = "attention"
    # Free-form override; if set, used instead of the preset. Comma-joined
    # suffixes matched against fully-qualified module names.
    target_modules: tuple[str, ...] = ()
    base_model: str = "ideogram-ai/ideogram-4-nf4"
    resolution: int = 1024

    def resolved_targets(self) -> tuple[str, ...]:
        if self.target_modules:
            return tuple(self.target_modules)
        if self.target_preset not in TARGET_PRESETS:
            raise ValueError(
                f"unknown target_preset={self.target_preset!r}; "
                f"choose from {sorted(TARGET_PRESETS)} or pass target_modules"
            )
        return TARGET_PRESETS[self.target_preset]


def _infer_features(base: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for nn.Linear / Linear4bit / Fp8Linear."""
    in_f = getattr(base, "in_features", None)
    out_f = getattr(base, "out_features", None)
    if in_f is not None and out_f is not None:
        return int(in_f), int(out_f)
    # Fall back to the weight shape (out, in) for plain linears.
    w = getattr(base, "weight", None)
    if w is not None and w.ndim == 2:
        return int(w.shape[1]), int(w.shape[0])
    raise TypeError(f"cannot infer in/out features for {type(base).__name__}")


class LoRALinear(nn.Module):
    """Wraps a frozen base linear and adds a trainable low-rank delta."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = _infer_features(base)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Adapter starts as a no-op (B=0), so generation matches the base model
        # exactly until training updates the weights.
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_B.weight)
        # LoRA params train in fp32 for stability even when the base is bf16/nf4.
        self.lora_A.to(torch.float32)
        self.lora_B.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x).to(torch.float32)))
        return out + (delta * self.scaling).to(out.dtype)


def _iter_target_modules(root: nn.Module, suffixes: tuple[str, ...]):
    for name, module in root.named_modules():
        if any(name.endswith(s) for s in suffixes):
            yield name, module


def inject_lora(transformer: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace each targeted linear in-place with a ``LoRALinear``.

    Freezes every base parameter first, then attaches adapters. Returns the
    list of fully-qualified module names that were wrapped.
    """
    for p in transformer.parameters():
        p.requires_grad_(False)

    suffixes = cfg.resolved_targets()
    wrapped: list[str] = []
    # Snapshot first: we mutate the module tree as we go.
    targets = list(_iter_target_modules(transformer, suffixes))
    for name, module in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = transformer.get_submodule(parent_path) if parent_path else transformer
        setattr(parent, attr, LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout))
        wrapped.append(name)
    if not wrapped:
        raise ValueError(
            f"no modules matched targets {suffixes!r}; run inspect_model.py to "
            f"list the real module names"
        )
    return wrapped


def lora_parameters(transformer: nn.Module):
    return [p for p in transformer.parameters() if p.requires_grad]


def count_parameters(transformer: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    return trainable, total


def lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors, keyed by their module path."""
    return {
        k: v.detach().cpu()
        for k, v in transformer.state_dict().items()
        if ".lora_A." in k or ".lora_B." in k
    }


def save_lora(transformer: nn.Module, cfg: LoRAConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_file(lora_state_dict(transformer), os.path.join(out_dir, "lora.safetensors"))
    with open(os.path.join(out_dir, "lora_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_lora_config(path: str) -> LoRAConfig:
    cfg_path = path if path.endswith(".json") else os.path.join(path, "lora_config.json")
    with open(cfg_path) as f:
        data = json.load(f)
    data["target_modules"] = tuple(data.get("target_modules", ()))
    return LoRAConfig(**data)


def apply_lora_checkpoint(transformer: nn.Module, path: str) -> LoRAConfig:
    """Inject adapters per the saved config, then load their weights."""
    cfg = load_lora_config(path)
    inject_lora(transformer, cfg)
    weights = path if path.endswith(".safetensors") else os.path.join(path, "lora.safetensors")
    state = load_file(weights)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if ".lora_A." in k or ".lora_B." in k]
    if unexpected:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected[:8]}")
    loaded = {k for k in state}
    not_loaded = [k for k in loaded if k in missing]
    if not_loaded:
        raise RuntimeError(f"LoRA keys failed to load: {not_loaded[:8]}")
    return cfg
