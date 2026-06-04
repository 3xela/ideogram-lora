"""Generate images from Ideogram 4, with or without a trained LoRA.

  # base model only
  python sample.py --prompt "a brutalist poster that says NIGHT SIGNAL" \
      --output ./examples/outputs/base.png

  # with a LoRA
  python sample.py --lora ./runs/test_lora \
      --prompt "a brutalist poster that says NIGHT SIGNAL" \
      --output ./examples/outputs/lora.png

  # before/after in one shot (writes base.png, lora.png, comparison_grid.png)
  python sample.py --lora ./runs/test_lora --compare \
      --prompt "a brutalist poster that says NIGHT SIGNAL" \
      --output ./examples/outputs/night_signal
"""

from __future__ import annotations

import argparse
import os

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    ap.add_argument("--prompt", required=True, help="plain text or JSON caption")
    ap.add_argument("--lora", default=None, help="path to a trained LoRA dir/safetensors")
    ap.add_argument("--output", required=True, help="png path, or basename when --compare")
    ap.add_argument("--compare", action="store_true", help="render base AND lora + a grid")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return ap.parse_args()


def generate(pipe, args):
    return pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        raise_on_caption_issues=False,
    )[0]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    print(f"Loading {args.model} (gated; downloads on first run) ...")
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=args.model),
        device=device,
        dtype=dtype,
    )

    if args.compare:
        from make_grid import grid_from_images

        out_base = args.output[:-4] if args.output.endswith(".png") else args.output
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

        print("Generating BASE ...")
        base_img = generate(pipe, args)
        base_path = f"{out_base}_base.png"
        base_img.save(base_path)

        if not args.lora:
            raise SystemExit("--compare needs --lora to compare against")
        from lora import apply_lora_checkpoint

        print(f"Applying LoRA {args.lora} ...")
        apply_lora_checkpoint(pipe.conditional_transformer, args.lora)
        print("Generating LoRA ...")
        lora_img = generate(pipe, args)
        lora_path = f"{out_base}_lora.png"
        lora_img.save(lora_path)

        grid = grid_from_images([base_img, lora_img], ["Base", "LoRA"])
        grid_path = f"{out_base}_comparison_grid.png"
        grid.save(grid_path)
        print(f"Saved:\n  {base_path}\n  {lora_path}\n  {grid_path}")
        return

    if args.lora:
        from lora import apply_lora_checkpoint

        print(f"Applying LoRA {args.lora} ...")
        apply_lora_checkpoint(pipe.conditional_transformer, args.lora)

    img = generate(pipe, args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
