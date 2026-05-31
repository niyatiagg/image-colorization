"""Build a 3x4 labeled grid showing the ControlNet zebra colorization progression.

Layout:
    cell (0,0) = grayscale input
    cells (0,1)..(2,2) = predictions from epochs 1..10
    cell (2,3) = ground-truth color image

Each preview_epoch_XXX.png is itself a 2-row by 3-column torchvision make_grid of
[cond | pred | target] (image_size=384, padding=2). The zebra is the top row;
columns 0/1/2 are cond/pred/target.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PREVIEW_DIR = Path("runs/controlnet_softedge")
OUT_PATH = Path("figures/zebra_progression.png")

EPOCHS = list(range(1, 11))
IMG = 384
PAD = 2

# Top row of make_grid corresponds to sample 0 (the zebra).
TOP = (PAD, PAD + IMG)  # y range
COND_X = (PAD, PAD + IMG)
PRED_X = (PAD + IMG + PAD, PAD + IMG + PAD + IMG)
TARG_X = (PAD + IMG + PAD + IMG + PAD, PAD + IMG + PAD + IMG + PAD + IMG)


def crop(im: Image.Image, x_range, y_range) -> Image.Image:
    x0, x1 = x_range
    y0, y1 = y_range
    return im.crop((x0, y0, x1, y1))


def to_grayscale_rgb(im: Image.Image) -> Image.Image:
    """ITU-R BT.601 luminance, replicated to 3 channels (matches the gray
    conditioning the script uses in inference)."""
    arr = np.asarray(im).astype(np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gray = np.stack([y, y, y], axis=-1)
    return Image.fromarray((gray * 255).clip(0, 255).astype(np.uint8))


def labeled_cell(im: Image.Image, label: str, cell_w: int, cell_h: int, font: ImageFont.FreeTypeFont,
                 label_h: int) -> Image.Image:
    img_h = cell_h - label_h
    im_resized = im.resize((cell_w, img_h), Image.BICUBIC)

    canvas = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
    canvas.paste(im_resized, (0, 0))

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (cell_w - tw) // 2
    ty = img_h + (label_h - th) // 2 - bbox[1]
    draw.text((tx, ty), label, fill=(20, 20, 20), font=font)
    return canvas


def main() -> None:
    preview_paths = {e: PREVIEW_DIR / f"preview_epoch_{e:03d}.png" for e in EPOCHS}
    missing = [str(p) for p in preview_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preview PNGs: {missing}")

    last_im = Image.open(preview_paths[EPOCHS[-1]]).convert("RGB")
    target_zebra = crop(last_im, TARG_X, TOP)
    gray_zebra = to_grayscale_rgb(target_zebra)

    predictions = []
    for e in EPOCHS:
        im = Image.open(preview_paths[e]).convert("RGB")
        predictions.append(crop(im, PRED_X, TOP))

    cells = [("Input (B&W)", gray_zebra)]
    for e, pred in zip(EPOCHS, predictions):
        cells.append((f"Epoch {e}", pred))
    cells.append(("Ground truth", target_zebra))

    assert len(cells) == 12, f"expected 12 cells, got {len(cells)}"

    rows, cols = 3, 4
    cell_w = 320
    cell_img_h = int(cell_w * (IMG / IMG))  # square images
    label_h = 36
    cell_h = cell_img_h + label_h
    margin = 16
    gutter = 8

    grid_w = cols * cell_w + (cols - 1) * gutter + 2 * margin
    grid_h = rows * cell_h + (rows - 1) * gutter + 2 * margin

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    font = ImageFont.truetype(font_path, 22)

    for idx, (label, im) in enumerate(cells):
        r = idx // cols
        c = idx % cols
        cell = labeled_cell(im, label, cell_w, cell_h, font, label_h)
        x = margin + c * (cell_w + gutter)
        y = margin + r * (cell_h + gutter)
        grid.paste(cell, (x, y))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT_PATH, "PNG", optimize=True)
    print(f"Saved {OUT_PATH}  ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
