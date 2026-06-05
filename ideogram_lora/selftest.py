"""No-weights self-test for the core mechanics.

Runs on CPU with only torch + safetensors installed. It does NOT touch the real
Ideogram 4 weights (gated, ~20GB). It builds a tiny stand-in transformer whose
submodule names match the real model (``layers.{i}.attention.qkv`` / ``.o`` and
``layers.{i}.feed_forward.w{1,2,3}``) and checks that:

  1. patchify / unpatchify round-trip exactly (the latent packing math),
  2. LoRA injects onto the right modules,
  3. the base is frozen and only LoRA params are trainable,
  4. a flow-matching step backprops and updates *only* the adapter,
  5. save -> reload reproduces the adapter outputs.

If this passes, the training/inference scripts are wired correctly; the only
unverified piece is loading the real gated checkpoint.
"""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn

from . import flow_utils
from .lora import (
    LoRAConfig,
    LoRALinear,
    LoRASlicedLinear,
    apply_lora_checkpoint,
    count_parameters,
    inject_lora,
    lora_parameters,
    save_lora,
)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.o(q + k + v)


class _MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attention = _Attn(d)
        self.feed_forward = _MLP(d, h)

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class _TinyTransformer(nn.Module):
    """Mirrors the real module-name layout at toy scale."""

    def __init__(self, d=32, h=64, n=3, out_ch=128):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d, h) for _ in range(n)])
        self.head = nn.Linear(d, out_ch)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_patchify_roundtrip():
    print("patchify round-trip")
    x = torch.randn(2, 32, 16, 24)  # (B, C, H, W)
    z = flow_utils.patchify(x, patch=2)
    check("packs to (B, L, 128)", z.shape == (2, (16 // 2) * (24 // 2), 2 * 2 * 32))
    back = flow_utils.unpatchify(z, gh=8, gw=12, patch=2)
    check("unpatchify is exact inverse", torch.allclose(back, x, atol=1e-6))


def test_flow_targets():
    print("flow-matching targets")
    clean = torch.randn(4, 10, 128)
    t = flow_utils.sample_timesteps(4, torch.device("cpu"))
    check("t in (0,1)", bool((t > 0).all() and (t < 1).all()))
    x_t, vel = flow_utils.make_flow_targets(clean, t)
    check("x_t shape matches", x_t.shape == clean.shape)
    # At t->1, x_t -> clean and velocity = clean - noise must satisfy the line.
    t_ = t.view(-1, 1, 1)
    recon_noise = (x_t - t_ * clean) / (1 - t_)
    check("path is consistent", torch.allclose(clean - recon_noise, vel, atol=1e-4))


def test_lora_inject_and_train():
    print("LoRA injection + training step")
    torch.manual_seed(0)
    model = _TinyTransformer()
    cfg = LoRAConfig(rank=4, alpha=8, target_preset="attention_mlp")
    wrapped = inject_lora(model, cfg)
    check("wrapped 3 blocks x 5 linears = 15", len(wrapped) == 15)
    check(
        "targets are the real families",
        all(
            w.endswith(("attention.qkv", "attention.o", "feed_forward.w1",
                        "feed_forward.w2", "feed_forward.w3"))
            for w in wrapped
        ),
    )
    check("LoRA modules in place", isinstance(model.layers[0].attention.qkv, LoRALinear))

    trainable, total = count_parameters(model)
    check("some params trainable", trainable > 0)
    check("most params frozen", trainable < total * 0.5)
    # Base weights must be frozen.
    base_frozen = all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if ".lora_A." not in n and ".lora_B." not in n
    )
    check("every base param frozen", base_frozen)

    # Adapter is a no-op at init (B=0) -> output equals a fresh base forward.
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        before = model(x).clone()

    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    target = torch.randn_like(before)
    base_snapshot = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if ".lora_" not in n
    }
    last = None
    for _ in range(5):
        opt.zero_grad()
        loss = flow_utils.flow_loss(model(x), target)
        loss.backward()
        opt.step()
        last = loss.item()
    check("loss is finite", last == last and abs(last) < 1e9)

    # Base unchanged, output moved (adapter learned something).
    base_unchanged = all(
        torch.equal(base_snapshot[n], p)
        for n, p in model.named_parameters()
        if ".lora_" not in n
    )
    check("base weights unchanged after training", base_unchanged)
    with torch.no_grad():
        after = model(x)
    check("adapter changed the output", not torch.allclose(before, after, atol=1e-5))

    # Save -> reload onto a fresh base reproduces the trained output.
    with tempfile.TemporaryDirectory() as d:
        save_lora(model, cfg, d)
        torch.manual_seed(0)
        fresh = _TinyTransformer()
        apply_lora_checkpoint(fresh, d)
        with torch.no_grad():
            reloaded = fresh(x)
        check("reloaded adapter matches", torch.allclose(after, reloaded, atol=1e-5))


def test_qkv_slice_targeting():
    print("fused-qkv slice targeting (Q/K/V)")
    torch.manual_seed(0)
    d = 32
    # The fused qkv emits [Q | K | V] as equal thirds of width d each.
    base = nn.Linear(d, 3 * d, bias=False)
    m = LoRASlicedLinear(base, rank=4, alpha=8, dropout=0.0, spans={"q": (0, d), "v": (2 * d, 3 * d)})
    x = torch.randn(2, 5, d)

    check("adapts only the requested spans", set(m.slices.keys()) == {"q", "v"})
    check("no-op at init (B=0)", torch.allclose(m(x), base(x), atol=1e-6))

    # Simulate a trained adapter: give the B matrices weight.
    for name in ("q", "v"):
        nn.init.normal_(m.slices[name]["lora_B"].weight, std=0.1)
    diff = (m(x) - base(x)).abs()
    check("Q span (0:d) adapted", diff[..., 0:d].max() > 1e-4)
    check("V span (2d:3d) adapted", diff[..., 2 * d : 3 * d].max() > 1e-4)
    check("K span (d:2d) untouched", torch.allclose(diff[..., d : 2 * d], torch.zeros(()), atol=1e-6))

    # Inject into the tiny transformer via target tokens, train, save -> reload.
    torch.manual_seed(0)
    model = _TinyTransformer()
    cfg = LoRAConfig(rank=4, alpha=8, target_modules=("attention.q", "attention.v", "attention.o"))
    wrapped = inject_lora(model, cfg)
    # 3 layers x (1 sliced qkv + 1 o) = 6 wrapped modules.
    check("wrapped sliced qkv + o per layer", len(wrapped) == 6)
    check("qkv became a sliced adapter", isinstance(model.layers[0].attention.qkv, LoRASlicedLinear))
    check("o is a plain LoRALinear", isinstance(model.layers[0].attention.o, LoRALinear))

    inp = torch.randn(2, 7, 32)
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    target = torch.randn(2, 7, 128)
    for _ in range(3):
        opt.zero_grad()
        flow_utils.flow_loss(model(inp), target).backward()
        opt.step()
    with torch.no_grad():
        trained = model(inp)

    with tempfile.TemporaryDirectory() as dtmp:
        save_lora(model, cfg, dtmp)
        torch.manual_seed(0)
        fresh = _TinyTransformer()
        apply_lora_checkpoint(fresh, dtmp)
        with torch.no_grad():
            reloaded = fresh(inp)
    check("sliced adapter reload matches", torch.allclose(trained, reloaded, atol=1e-5))


def test_layer_targeting():
    print("per-block layer targeting")
    from .lora import LoRASlicedLinear  # noqa: F401 (already imported above)

    # Tiny transformer has 3 blocks (layers.0/1/2).
    torch.manual_seed(0)
    model = _TinyTransformer()
    cfg = LoRAConfig(rank=4, alpha=8, target_modules=("attention.q", "attention.o"), layers=(0, 2))
    wrapped = inject_lora(model, cfg)
    # 2 chosen blocks x (qkv slice + o) = 4; block 1 must be left alone.
    check("only chosen blocks wrapped", len(wrapped) == 4)
    check("block 0 qkv adapted", isinstance(model.layers[0].attention.qkv, LoRASlicedLinear))
    check("block 1 qkv untouched", not isinstance(model.layers[1].attention.qkv, LoRASlicedLinear))
    check("block 2 o adapted", isinstance(model.layers[2].attention.o, LoRALinear))
    check("block 1 o untouched", not isinstance(model.layers[1].attention.o, LoRALinear))


def main() -> None:
    test_patchify_roundtrip()
    test_flow_targets()
    test_lora_inject_and_train()
    test_qkv_slice_targeting()
    test_layer_targeting()
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
