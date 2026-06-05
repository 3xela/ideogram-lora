"""Load Ideogram 4 and print its real structure + LoRA target candidates.

Run this first. It confirms your gated HF access works and shows the actual
module names (which are NOT the diffusers-style ``to_q/to_k/...`` you may expect
from other ecosystems).

    python inspect_model.py
    python inspect_model.py --model ideogram-ai/ideogram-4-nf4 --device cuda
"""

from __future__ import annotations

import argparse
import collections

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--full", action="store_true", help="print every linear, not just a sample"
    )
    args = ap.parse_args()

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    from .fast_init import no_init_weights

    cfg = Ideogram4PipelineConfig(weights_repo=args.model)
    print(f"Loading {args.model} (this downloads gated weights on first run)...")
    with no_init_weights():  # the checkpoint overwrites every weight; skip the slow RNG init
        pipe = Ideogram4Pipeline.from_pretrained(
            config=cfg, device=args.device, dtype=getattr(torch, args.dtype)
        )

    print("\nPipeline components:")
    for name in (
        "conditional_transformer",
        "unconditional_transformer",
        "text_encoder",
        "text_tokenizer",
        "autoencoder",
    ):
        comp = getattr(pipe, name, None)
        print(f"  - {name}: {type(comp).__name__}")

    tf = pipe.conditional_transformer
    print(f"\nconditional_transformer: {type(tf).__name__}")
    print(f"  layers: {len(tf.layers)}  emb_dim: {tf.config.emb_dim}  heads: {tf.config.num_heads}")

    # Collect linear-like leaves by their suffix family.
    families: dict[str, list[str]] = collections.defaultdict(list)
    for name, module in tf.named_modules():
        if hasattr(module, "weight") and getattr(module, "weight", None) is not None:
            w = module.weight
            if getattr(w, "ndim", 0) == 2 or "Linear" in type(module).__name__:
                # Family = the trailing two path components, e.g. "attention.qkv".
                fam = ".".join(name.split(".")[-2:])
                families[fam].append(name)

    print("\nLinear module families in conditional_transformer:")
    for fam, names in sorted(families.items()):
        print(f"  {fam:<24} x{len(names)}  e.g. {names[0]}")

    print("\nRecommended LoRA targets (start conservative):")
    print("  --target_modules attention        -> attention.qkv, attention.o")
    print("  --target_modules attention_mlp     -> + feed_forward.w1/w2/w3")
    print("  --target_modules all_linear        -> same as attention_mlp here")

    if args.full:
        print("\nAll candidate linears:")
        for fam, names in sorted(families.items()):
            for n in names:
                print(f"  {n}")


if __name__ == "__main__":
    main()
