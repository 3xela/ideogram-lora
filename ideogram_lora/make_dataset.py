"""Generate a consistent claymation-penguin training set with the base model.

Uses Ideogram 4's native JSON caption format (the model is trained on it, so it
stays in-distribution: higher quality AND it avoids the gray "Image blocked by
safety filter" placeholder that out-of-distribution plain-text prompts trip).

Two captions per scene:
  - GENERATION caption: styled JSON (claymation style_description) -> renders clay.
  - TRAINING caption (.txt): style-FREE JSON describing the same penguin/scene
    with no clay words, so the LoRA binds the clay look to the concept itself.
    A bare penguin prompt should then come out claymation after training.

    python make_dataset.py --out ./dataset_penguin --steps 56
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

# Detail metric: the blocked placeholder is near-flat (variance-of-Laplacian <=33),
# real renders are >=78. Threshold 50 separates them; retry a fresh seed if tripped.
BLOCK_LAPVAR_THRESHOLD = 50.0

# Claymation style, applied only to GENERATION captions (kept out of training).
STYLE_DESC = {
    "aesthetics": "claymation stop-motion, handmade plasticine miniature, charming",
    "lighting": "soft diffused studio light, cool-neutral white balance",
    "photo": "macro tilt-shift studio photograph, shallow depth of field",
    "medium": "Photograph",
    "color_palette": ["#e8eef4", "#1d1d1d", "#f4f4f0", "#e8902a"],
}

CLAY_PENGUIN = (
    "A small plasticine clay penguin, glossy black back and head, off-white "
    "belly, orange beak and feet, visible fingerprints and sculpt marks in the "
    "clay. {pose}"
)
PLAIN_PENGUIN = (
    "A penguin, black back and head, white belly, orange beak and webbed feet. "
    "{pose}"
)

# (filename idx, high-level subject, pose, background shell, seed)
SCENES = [
    ("a penguin standing on a snowy iceberg", "Standing upright, facing forward.",
     "A snowy iceberg surface receding into a softly blurred cool-grey backdrop.", 11),
    ("a penguin mid-waddle on pale blue ice", "Leaning forward mid-waddle, flippers held slightly out.",
     "Pale blue ice surface receding into a softly blurred cool-grey backdrop.", 22),
    ("a penguin sitting on a smooth grey rock", "Sitting low on its belly, head up.",
     "A smooth grey rock surface with a softly blurred cool backdrop.", 33),
    ("a penguin looking up with both flippers raised", "Head tilted up, both flippers raised outward.",
     "Pale snow-dusted ground receding into a softly blurred cool-grey backdrop.", 44),
    ("a penguin beside a small fish by the water", "Standing upright beside a small fish, looking down at it.",
     "Wet pebbled shoreline by calm water, softly blurred cool backdrop.", 55),
    ("a penguin chick under gently falling snow", "Round fluffy chick standing upright, looking forward.",
     "Soft snow-covered ground with faint falling snow, blurred cool-grey backdrop.", 66),
]


def _caption(hld: str, background: str, obj_desc: str, styled: bool) -> str:
    cap = {"high_level_description": hld}
    if styled:
        cap["style_description"] = STYLE_DESC
    cap["compositional_deconstruction"] = {
        "background": background,
        "elements": [{"type": "obj", "desc": obj_desc}],
    }
    return json.dumps(cap, ensure_ascii=False, separators=(",", ":"))


def gen_caption(subject: str, pose: str, background: str) -> str:
    hld = f"A claymation stop-motion photograph of {subject}, miniature handmade diorama."
    return _caption(hld, background, CLAY_PENGUIN.format(pose=pose), styled=True)


def train_caption(subject: str, pose: str, background: str) -> str:
    hld = f"A photograph of {subject}."
    return _caption(hld, background, PLAIN_PENGUIN.format(pose=pose), styled=False)


def detail_score(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float64)
    lap = -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    return float(lap[1:-1, 1:-1].var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./dataset_penguin")
    ap.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--max_retries", type=int, default=6,
                    help="seed retries if a render trips the safety placeholder")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=args.model),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    for i, (subject, pose, background, seed) in enumerate(SCENES, start=1):
        gen = gen_caption(subject, pose, background)
        print(f"\n[{i}/{len(SCENES)}] {subject}", flush=True)

        best_img, best_score = None, -1.0
        for attempt in range(args.max_retries):
            try_seed = seed + attempt * 1000
            img = pipe(gen, height=args.resolution, width=args.resolution,
                       num_steps=args.steps, guidance_scale=args.guidance,
                       seed=try_seed, raise_on_caption_issues=False)[0]
            score = detail_score(img)
            if score > best_score:
                best_img, best_score = img, score
            if score >= BLOCK_LAPVAR_THRESHOLD:
                print(f"  seed={try_seed} ok (detail={score:.0f})", flush=True)
                break
            print(f"  seed={try_seed} BLOCKED (detail={score:.0f}), retrying", flush=True)
        else:
            print(f"  ! all {args.max_retries} attempts blocked; keeping best "
                  f"(detail={best_score:.0f})", flush=True)

        stem = out / f"{i:04d}"
        best_img.save(stem.with_suffix(".png"))
        # Training caption is style-FREE JSON so the clay look binds to the concept.
        stem.with_suffix(".txt").write_text(
            train_caption(subject, pose, background) + "\n", encoding="utf-8")
        print(f"  saved {stem.with_suffix('.png')}", flush=True)

    print(f"\nDataset ready in {out} ({len(SCENES)} pairs)")


if __name__ == "__main__":
    main()
