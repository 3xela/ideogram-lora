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
import re
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


# Real module families in Ideogram4Transformer.layers[i] (verified against
# ideogram4.modeling_ideogram4). Note the *fused* qkv and the o projection.
#
# The attention projection is a single fused linear `attention.qkv` of width
# 3*hidden emitting [Q | K | V] as equal contiguous thirds. The slice tokens
# `attention.q`/`.k`/`.v` adapt one of those thirds individually (see
# LoRASlicedLinear); `attention.qkv` adapts all three at once.
TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "attention": ("attention.qkv", "attention.o"),
    "attention_qv": ("attention.q", "attention.v", "attention.o"),
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

# Slice tokens -> name of the span inside the fused qkv they adapt.
QKV_SLICE_TOKENS: dict[str, str] = {
    "attention.q": "q",
    "attention.k": "k",
    "attention.v": "v",
}


def _qkv_spans(out_features: int) -> dict[str, tuple[int, int]]:
    """(lo, hi) output ranges of q/k/v inside a fused qkv linear.

    The fused projection emits [Q | K | V] as equal thirds (q, k, v all =
    hidden_size; the model has no grouped-query attention).
    """
    if out_features % 3 != 0:
        raise ValueError(f"fused qkv out_features={out_features} is not divisible by 3")
    h = out_features // 3
    return {"q": (0, h), "k": (h, 2 * h), "v": (2 * h, 3 * h)}


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_preset: str = "attention"
    # Free-form override; if set, used instead of the preset. Suffixes matched
    # against fully-qualified module names (e.g. "attention.o", "feed_forward.w1"),
    # plus the fused-qkv slice tokens "attention.q"/".k"/".v" that adapt a single
    # projection inside the shared qkv linear.
    target_modules: tuple[str, ...] = ()
    # Block indices to adapt (e.g. (0, 1, 2, 30, 31)); empty = every layer.
    # Filters which transformer blocks get adapters; target_modules picks which
    # projections inside each chosen block.
    layers: tuple[int, ...] = ()
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
        # LoRA params train in fp32 for stability even when the base is bf16/nf4,
        # but must live on the same device as the (frozen) base layer.
        base_device = next(self.base.parameters()).device
        self.lora_A.to(device=base_device, dtype=torch.float32)
        self.lora_B.to(device=base_device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x).to(torch.float32)))
        return out + (delta * self.scaling).to(out.dtype)


class LoRASlicedLinear(nn.Module):
    """Wraps a fused linear (the attention qkv) and adds an independent low-rank
    delta to one or more contiguous output spans, leaving the rest untouched.

    Used to adapt Q, K and/or V individually even though they share a single
    fused projection: each requested span gets its own A/B pair, and the deltas
    are zero-padded back to the full width before being added to the base output.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
        spans: dict[str, tuple[int, int]],
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = _infer_features(base)
        self.out_features = out_f
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.slices = nn.ModuleDict()
        self.spans: dict[str, tuple[int, int]] = dict(spans)
        base_device = next(self.base.parameters()).device
        for name, (lo, hi) in spans.items():
            a = nn.Linear(in_f, rank, bias=False)
            b = nn.Linear(rank, hi - lo, bias=False)
            nn.init.normal_(a.weight, std=1.0 / rank)
            nn.init.zeros_(b.weight)  # no-op adapter at init, like LoRALinear
            a.to(device=base_device, dtype=torch.float32)
            b.to(device=base_device, dtype=torch.float32)
            self.slices[name] = nn.ModuleDict({"lora_A": a, "lora_B": b})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xd = self.dropout(x).to(torch.float32)
        for name, (lo, hi) in self.spans.items():
            ad = self.slices[name]
            delta = ad["lora_B"](ad["lora_A"](xd))
            delta = F.pad(delta, (lo, self.out_features - hi))  # scatter into the span
            out = out + (delta * self.scaling).to(out.dtype)
        return out


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    """Block index from a module name like 'layers.7.attention.qkv', else None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _iter_target_modules(root: nn.Module, suffixes: tuple[str, ...], layers=None):
    keep = set(layers) if layers else None
    for name, module in root.named_modules():
        if not any(name.endswith(s) for s in suffixes):
            continue
        if keep is not None and _layer_index(name) not in keep:
            continue
        yield name, module


def _replace(transformer: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_path, _, attr = name.rpartition(".")
    parent = transformer.get_submodule(parent_path) if parent_path else transformer
    setattr(parent, attr, new_module)


def inject_lora(transformer: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace each targeted linear in-place with a LoRA wrapper.

    Full-module targets (``attention.o``, ``feed_forward.*``, ``attention.qkv``)
    become ``LoRALinear``; the fused-qkv slice tokens (``attention.q``/``.k``/
    ``.v``) wrap the shared qkv linear in a single ``LoRASlicedLinear`` adapting
    just those spans. Freezes every base parameter first. Returns the list of
    fully-qualified module names that were wrapped.
    """
    for p in transformer.parameters():
        p.requires_grad_(False)

    tokens = cfg.resolved_targets()
    slice_names = [QKV_SLICE_TOKENS[t] for t in tokens if t in QKV_SLICE_TOKENS]
    full_suffixes = tuple(t for t in tokens if t not in QKV_SLICE_TOKENS)
    if slice_names and "attention.qkv" in full_suffixes:
        raise ValueError(
            "target the fused 'attention.qkv' or its slices "
            "('attention.q'/'.k'/'.v'), not both"
        )

    wrapped: list[str] = []
    # Snapshot matches first: we mutate the module tree as we go.
    if full_suffixes:
        for name, module in list(_iter_target_modules(transformer, full_suffixes, cfg.layers)):
            _replace(transformer, name, LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout))
            wrapped.append(name)
    if slice_names:
        for name, module in list(_iter_target_modules(transformer, ("attention.qkv",), cfg.layers)):
            spans = _qkv_spans(_infer_features(module)[1])
            chosen = {n: spans[n] for n in slice_names}
            _replace(
                transformer, name, LoRASlicedLinear(module, cfg.rank, cfg.alpha, cfg.dropout, chosen)
            )
            wrapped.append(f"{name}[{'+'.join(slice_names)}]")

    if not wrapped:
        scope = f" within layers {tuple(cfg.layers)}" if cfg.layers else ""
        raise ValueError(
            f"no modules matched targets {tokens!r}{scope}; run inspect_model.py "
            f"to list the real module names"
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
    data["layers"] = tuple(data.get("layers", ()))
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
