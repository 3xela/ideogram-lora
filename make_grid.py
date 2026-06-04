"""Stitch images into a labelled comparison grid. No fancy evaluation.

  python make_grid.py --images base.png lora.png --labels Base LoRA \
      --output comparison_grid.png
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grid_from_images(
    images: list[Image.Image],
    labels: list[str] | None = None,
    *,
    pad: int = 12,
    label_h: int = 40,
    bg: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    """Lay images out in a single row with optional labels above each."""
    images = [im.convert("RGB") for im in images]
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    labels = labels or [""] * len(images)

    n = len(images)
    W = pad + n * (cell_w + pad)
    H = pad + label_h + cell_h + pad
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    font = _font(24)

    x = pad
    for im, label in zip(images, labels):
        if label:
            draw.text((x + 4, pad), label, fill=(20, 20, 20), font=font)
        canvas.paste(im, (x, pad + label_h))
        x += cell_w + pad
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    images = [Image.open(p) for p in args.images]
    labels = args.labels if args.labels else [f"{i}" for i in range(len(images))]
    if len(labels) != len(images):
        raise SystemExit("--labels count must match --images count")
    grid_from_images(images, labels).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
