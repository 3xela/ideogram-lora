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
import json
import os
from pathlib import Path

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


def resolve_caption(prompt_arg: str) -> str:
    """Turn ``--prompt`` (raw string, or a path to a .json/.txt file) into the
    caption string fed to the model.

    Ideogram 4 is trained on single-line JSON captions, so we prefer them: JSON
    is normalized (``aspect_ratio`` dropped, the model never sees it, and keys
    reordered to the schema) and validated with the package's CaptionVerifier.
    Plain text still passes through, with a note that it's out-of-distribution
    (and thus likelier to trip the gray safety placeholder).
    """
    text = prompt_arg
    p = Path(prompt_arg)
    if p.exists() and p.suffix.lower() in (".json", ".txt"):
        text = p.read_text(encoding="utf-8")
    text = text.strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        print("  prompt: plain text (tip: JSON captions are in-distribution and "
              "less likely to trip the safety placeholder)")
        return text
    if not isinstance(obj, dict):
        return text

    from ideogram4.caption_verifier import CaptionVerifier
    from ideogram4.magic_prompt import reorder_caption_keys

    obj.pop("aspect_ratio", None)
    obj = reorder_caption_keys(obj)
    warnings = CaptionVerifier().verify(obj)
    if warnings:
        print("  caption warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  prompt: valid JSON caption")
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def generate(pipe, caption, args):
    return pipe(
        caption,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        raise_on_caption_issues=False,
    )[0]


def main() -> None:
    args = parse_args()
    caption = resolve_caption(args.prompt)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    from .fast_init import no_init_weights

    print(f"Loading {args.model} (gated; downloads on first run) ...")
    with no_init_weights():  # the checkpoint overwrites every weight; skip the slow RNG init
        pipe = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=args.model),
            device=device,
            dtype=dtype,
        )

    if args.compare:
        from .make_grid import grid_from_images

        out_base = args.output[:-4] if args.output.endswith(".png") else args.output
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

        print("Generating BASE ...")
        base_img = generate(pipe, caption, args)
        base_path = f"{out_base}_base.png"
        base_img.save(base_path)

        if not args.lora:
            raise SystemExit("--compare needs --lora to compare against")
        from .lora import apply_lora_checkpoint

        print(f"Applying LoRA {args.lora} ...")
        apply_lora_checkpoint(pipe.conditional_transformer, args.lora)
        print("Generating LoRA ...")
        lora_img = generate(pipe, caption, args)
        lora_path = f"{out_base}_lora.png"
        lora_img.save(lora_path)

        grid = grid_from_images([base_img, lora_img], ["Base", "LoRA"])
        grid_path = f"{out_base}_comparison_grid.png"
        grid.save(grid_path)
        print(f"Saved:\n  {base_path}\n  {lora_path}\n  {grid_path}")
        return

    if args.lora:
        from .lora import apply_lora_checkpoint

        print(f"Applying LoRA {args.lora} ...")
        apply_lora_checkpoint(pipe.conditional_transformer, args.lora)

    img = generate(pipe, caption, args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
