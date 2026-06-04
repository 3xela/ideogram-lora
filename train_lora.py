"""Train a LoRA adapter on Ideogram 4 from local image+caption pairs.

Pipeline:
  1. Load Ideogram 4 (gated weights via your HF token).
  2. Pre-encode every sample once: image -> VAE latent, caption -> Qwen3-VL
     features. Cached on CPU so the text encoder, VAE, and the unconditional
     transformer can be freed before training (this is what keeps it on a
     single ~40GB GPU).
  3. Inject LoRA into the conditional transformer, freeze everything else.
  4. Flow-matching loss, AdamW on the adapter only.
  5. Save lora.safetensors + lora_config.json.

Example:
  python train_lora.py \
    --model ideogram-ai/ideogram-4-nf4 \
    --data ./dataset_example \
    --output ./runs/test_lora \
    --resolution 1024 --rank 16 --learning_rate 1e-4 --max_train_steps 500

These defaults are experimental. If you OOM, drop --resolution to 512 first.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.utils.checkpoint as cp
from PIL import Image
from tqdm import tqdm

import flow_utils
from lora import (
    LoRAConfig,
    count_parameters,
    inject_lora,
    lora_parameters,
    save_lora,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    ap.add_argument("--data", required=True, help="folder of <name>.<img> + <name>.txt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--target_modules",
        default="attention",
        help="preset (attention|attention_mlp|all_linear) or comma list of suffixes",
    )
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_train_steps", type=int, default=500)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--checkpoint_every", type=int, default=250)
    ap.add_argument("--t_sample_mean", type=float, default=0.0)
    ap.add_argument("--t_sample_std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--vae_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample the VAE posterior (vs. use the mean) when encoding latents",
    )
    return ap.parse_args()


def discover_pairs(data_dir: str) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for img in sorted(Path(data_dir).iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = img.with_suffix(".txt")
        if not cap.exists():
            print(f"  ! skipping {img.name}: no matching {cap.name}")
            continue
        text = cap.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  ! skipping {img.name}: empty caption")
            continue
        pairs.append((img, text))
    return pairs


def load_image_tensor(path: Path, res: int, device, dtype) -> torch.Tensor:
    """Center-crop to square, resize to res, return (1, 3, res, res) in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    img = img.resize((res, res), Image.LANCZOS)
    t = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device=device, dtype=dtype)


def _to_array(img: "Image.Image"):
    import numpy as np

    return np.asarray(img, dtype="uint8")


def enable_grad_checkpointing(transformer) -> None:
    for blk in transformer.layers:
        if getattr(blk, "_ckpt_wrapped", False):
            continue
        orig = blk.forward

        def wrapper(*a, _orig=orig, **kw):
            return cp.checkpoint(_orig, *a, use_reentrant=False, **kw)

        blk.forward = wrapper
        blk._ckpt_wrapped = True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    patch_px = 16  # patch_size(2) * ae_scale_factor(8)
    if args.resolution % patch_px != 0:
        raise SystemExit(f"--resolution must be a multiple of {patch_px}")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"Discovering dataset in {args.data} ...")
    pairs = discover_pairs(args.data)
    if not pairs:
        raise SystemExit("no usable image+caption pairs found")
    print(f"  {len(pairs)} sample(s)")

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    print(f"Loading {args.model} (gated; downloads on first run) ...")
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=args.model),
        device=device,
        dtype=dtype,
    )

    # We only train the conditional branch; free the unconditional one now.
    pipe.unconditional_transformer = None
    torch.cuda.empty_cache()

    # ---- Pre-encode everything once, then free the heavy frozen encoders. ----
    print("Pre-encoding latents + text features (one-time) ...")
    cache: list[dict] = []
    for img_path, caption in tqdm(pairs):
        image = load_image_tensor(img_path, args.resolution, device, dtype)
        clean = flow_utils.encode_to_latent(
            pipe.autoencoder,
            image,
            pipe.latent_shift,
            pipe.latent_scale,
            sample=args.vae_sample,
        )  # (1, num_img, 128)

        inputs = pipe._build_inputs([caption], height=args.resolution, width=args.resolution)
        llm = pipe._encode_text(
            inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
        )  # (1, L, D)
        num_text = int(inputs["max_text_tokens"])
        num_img = int(inputs["num_image_tokens"])

        cache.append(
            {
                "clean": clean.squeeze(0).to("cpu", torch.bfloat16),
                "text_feats": llm[:, :num_text].squeeze(0).to("cpu", torch.bfloat16),
                "position_ids": inputs["position_ids"].squeeze(0).cpu(),
                "segment_ids": inputs["segment_ids"].squeeze(0).cpu(),
                "indicator": inputs["indicator"].squeeze(0).cpu(),
                "num_text": num_text,
                "num_img": num_img,
            }
        )

    feat_dim = cache[0]["text_feats"].shape[-1]
    pipe.text_encoder = None
    pipe.text_tokenizer = None
    pipe.autoencoder = None
    torch.cuda.empty_cache()

    # ---- Attach LoRA to the conditional transformer. ----
    tf = pipe.conditional_transformer
    targets = args.target_modules
    cfg = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_preset=targets if "." not in targets and "," not in targets else "attention",
        target_modules=tuple(targets.split(",")) if ("." in targets or "," in targets) else (),
        base_model=args.model,
        resolution=args.resolution,
    )
    wrapped = inject_lora(tf, cfg)
    print(f"Injected LoRA into {len(wrapped)} modules (targets: {cfg.resolved_targets()})")
    trainable, total = count_parameters(tf)
    print(f"  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    if args.gradient_checkpointing:
        enable_grad_checkpointing(tf)
        print("  gradient checkpointing: on")

    tf.train()
    opt = torch.optim.AdamW(lora_parameters(tf), lr=args.learning_rate)

    # ---- Training loop. ----
    order: list[int] = []
    accum = max(1, args.gradient_accumulation_steps)
    running = 0.0
    pbar = tqdm(range(args.max_train_steps), desc="train")
    opt.zero_grad(set_to_none=True)

    for step in pbar:
        for micro in range(accum):
            if not order:
                order = list(range(len(cache)))
                random.shuffle(order)
            s = cache[order.pop()]

            num_text, num_img = s["num_text"], s["num_img"]
            L = num_text + num_img
            clean = s["clean"].unsqueeze(0).to(device, torch.float32)  # (1, num_img, 128)
            text_feats = s["text_feats"].unsqueeze(0).to(device, dtype)
            position_ids = s["position_ids"].unsqueeze(0).to(device)
            segment_ids = s["segment_ids"].unsqueeze(0).to(device)
            indicator = s["indicator"].unsqueeze(0).to(device)

            llm_full = torch.zeros(1, L, feat_dim, device=device, dtype=dtype)
            llm_full[:, :num_text] = text_feats

            t = flow_utils.sample_timesteps(
                1, device, mean=args.t_sample_mean, std=args.t_sample_std
            )
            x_t, target = flow_utils.make_flow_targets(clean, t)

            x_full = torch.zeros(1, L, clean.shape[-1], device=device, dtype=torch.float32)
            x_full[:, num_text:] = x_t
            # With checkpointing, the only grad-requiring tensors inside a block
            # are the LoRA params; marking the input keeps recompute unambiguous.
            if args.gradient_checkpointing:
                x_full.requires_grad_(True)

            pred = tf(
                llm_features=llm_full,
                x=x_full,
                t=t,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )
            pred_img = pred[:, num_text:]
            loss = flow_utils.flow_loss(pred_img, target) / accum
            loss.backward()
            running += loss.item()

        torch.nn.utils.clip_grad_norm_(lora_parameters(tf), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        pbar.set_postfix(loss=f"{running:.4f}")
        running = 0.0

        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            ckpt = os.path.join(args.output, f"checkpoint-{step + 1}")
            save_lora(tf, cfg, ckpt)
            pbar.write(f"  saved {ckpt}")

    save_lora(tf, cfg, args.output)
    print(f"\nDone. LoRA saved to {args.output}")
    print(f"  {os.path.join(args.output, 'lora.safetensors')}")
    print(f"  {os.path.join(args.output, 'lora_config.json')}")


if __name__ == "__main__":
    main()
