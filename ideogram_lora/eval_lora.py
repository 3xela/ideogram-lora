"""Evaluate the penguin LoRA: base vs every checkpoint, on two prompts.

Row 1 = a training-like prompt (does the clay look appear?).
Row 2 = a HELD-OUT prompt not in the training set (does the style generalize,
or did it just memorize the 6 training scenes?).

Loads the base model once, renders the base row, injects the LoRA structure
once, then swaps in each checkpoint's weights (no 20GB reload per checkpoint).

    python eval_lora.py --run ./runs/penguin_clay --out ./examples/outputs/penguin_eval.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from .lora import apply_lora_checkpoint
from .make_grid import grid_from_images

# Style-FREE captions: the clay look must come from the LoRA, not the words.
PROMPTS = {
    "standing (train-like)": {
        "high_level_description": "A photograph of a penguin standing upright on snow.",
        "compositional_deconstruction": {
            "background": "Snow-covered ground receding into a softly blurred cool-grey backdrop.",
            "elements": [{"type": "obj", "desc": "A penguin, black back and head, white belly, orange beak and webbed feet, standing upright facing forward."}],
        },
    },
    "scarf (held-out)": {
        "high_level_description": "A photograph of a penguin wearing a tiny red knitted scarf, standing on snow.",
        "compositional_deconstruction": {
            "background": "Snow-covered ground receding into a softly blurred cool-grey backdrop.",
            "elements": [{"type": "obj", "desc": "A penguin, black back and head, white belly, orange beak and webbed feet, standing upright, wearing a small red knitted scarf around its neck."}],
        },
    },
}


def minify(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def discover_checkpoints(run: Path) -> list[tuple[str, Path]]:
    """Return [(label, dir)] for checkpoint-* dirs (numeric order)."""
    ckpts = []
    for d in run.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)", d.name)
        if m and (d / "lora.safetensors").exists():
            ckpts.append((int(m.group(1)), d))
    ckpts.sort()
    return [(f"step {n}", d) for n, d in ckpts]


def vstack(rows: list[Image.Image], pad: int = 12, bg=(245, 245, 245)) -> Image.Image:
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + pad * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + pad
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="./runs/penguin_clay")
    ap.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    ap.add_argument("--out", default="./examples/outputs/penguin_eval.png")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = Path(args.run)
    ckpts = discover_checkpoints(run)
    if not ckpts:
        raise SystemExit(f"no checkpoints found under {run}")
    print("checkpoints:", [c[0] for c in ckpts], flush=True)

    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=args.model),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    def gen(cap: str) -> Image.Image:
        return pipe(cap, height=args.res, width=args.res, num_steps=args.steps,
                    guidance_scale=args.guidance, seed=args.seed,
                    raise_on_caption_issues=False)[0]

    captions = {name: minify(obj) for name, obj in PROMPTS.items()}

    # 1) Base row (pristine transformer, before any injection).
    cols = {name: [gen(cap)] for name, cap in captions.items()}
    labels = ["base"]

    # 2) Inject once from the first checkpoint, then swap weights per checkpoint.
    tf = pipe.conditional_transformer
    for i, (label, ckpt) in enumerate(ckpts):
        if i == 0:
            apply_lora_checkpoint(tf, str(ckpt))
        else:
            state = load_file(str(ckpt / "lora.safetensors"))
            tf.load_state_dict(state, strict=False)
        labels.append(label)
        for name, cap in captions.items():
            cols[name].append(gen(cap))
        print(f"rendered {label}", flush=True)

    rows = [grid_from_images(cols[name], [f"{name} | {l}" for l in labels])
            for name in PROMPTS]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vstack(rows).save(out_path)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
