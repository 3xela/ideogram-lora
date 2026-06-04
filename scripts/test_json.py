"""Test whether proper Ideogram-4 JSON captions avoid the safety placeholder.

Renders the three scenes that blocked with plain-text prompts, at the SAME
seeds, but as native minified-JSON captions. If detail scores clear the
threshold, JSON in-distribution prompting is the fix.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

STYLE = {
    "aesthetics": "claymation stop-motion, handmade plasticine miniature, charming",
    "lighting": "soft diffused studio light, cool-neutral white balance",
    "photo": "macro tilt-shift studio photograph, shallow depth of field",
    "medium": "Photograph",
    "color_palette": ["#e8eef4", "#1d1d1d", "#f4f4f0", "#e8902a"],
}

PENGUIN = (
    "A small plasticine clay penguin, glossy black back and head, off-white "
    "belly, orange beak and feet, visible fingerprints and sculpt marks in the "
    "clay. {pose}"
)


def caption(hld: str, background: str, pose: str) -> str:
    obj = {"type": "obj", "desc": PENGUIN.format(pose=pose)}
    cap = {
        "high_level_description": hld,
        "style_description": STYLE,
        "compositional_deconstruction": {"background": background, "elements": [obj]},
    }
    return json.dumps(cap, ensure_ascii=False, separators=(",", ":"))


# (basename, seed, json caption): the three previously-blocked scenes.
CASES = [
    ("json_waddle", 22, caption(
        "A claymation stop-motion photograph of a plasticine penguin mid-waddle on a pale blue ice floe, miniature diorama.",
        "Pale blue ice surface receding into a softly blurred cool-grey studio backdrop.",
        "Leaning forward mid-waddle, flippers held slightly out from the body.")),
    ("json_flippers", 44, caption(
        "A claymation stop-motion photograph of a plasticine penguin looking up with both flippers raised, miniature diorama.",
        "Pale snow-dusted ground receding into a softly blurred cool-grey studio backdrop.",
        "Head tilted up, both flippers raised outward.")),
    ("json_chick", 66, caption(
        "A claymation stop-motion photograph of a small plasticine penguin chick under gently falling snow, miniature diorama.",
        "Soft snow-covered ground with faint falling snow, blurred cool-grey studio backdrop.",
        "Round fluffy chick standing upright, looking forward.")),
]


def detail(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float64)
    l = -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    return float(l[1:-1, 1:-1].var())


def main() -> None:
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-nf4"),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    for name, seed, cap in CASES:
        img = pipe(cap, height=1024, width=1024, num_steps=56,
                   guidance_scale=7.0, seed=seed, raise_on_caption_issues=False)[0]
        out = f"./examples/outputs/{name}.png"
        img.save(out)
        d = detail(img)
        print(f"{name}: seed={seed} detail={d:.0f} {'OK' if d >= 50 else 'BLOCKED'} -> {out}", flush=True)


if __name__ == "__main__":
    main()
