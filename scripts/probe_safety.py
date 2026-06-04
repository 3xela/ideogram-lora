"""Probe what trips the baked-in 'Image blocked by safety filter' behavior.

Loads the pipeline once, renders several prompts that isolate one variable
each, and saves them so we can eyeball which dimension triggers the block.
"""

from __future__ import annotations

import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

PROMPTS = [
    # (basename, prompt): each isolates one suspected trigger.
    ("p_dataset_nightsignal", 'a minimal brutalist black and white poster with the large distorted title "NIGHT SIGNAL", heavy uppercase type, high contrast, grainy texture'),
    ("p_text_only_hello", 'a poster with the word "HELLO" in bold black letters on a white background'),
    ("p_brutalist_building", "a photograph of a brutalist concrete building, overcast sky"),
    ("p_punk_no_text", "a grainy high-contrast black and white punk zine collage, photocopied texture, no text"),
]


def main() -> None:
    device = torch.device("cuda")
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-nf4"),
        device=device,
        dtype=torch.bfloat16,
    )
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===\n{prompt}", flush=True)
        img = pipe(
            prompt,
            height=1024,
            width=1024,
            num_steps=48,
            guidance_scale=7.0,
            seed=0,
            raise_on_caption_issues=False,
        )[0]
        out = f"./examples/outputs/{name}.png"
        img.save(out)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
