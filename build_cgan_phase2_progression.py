"""Build progression figures from cGAN phase-2 previews (first batch image only).

Preview layout (cgan_places365.save_preview, max_images=4): 4 rows × 3 columns
per epoch — each row is [grayscale L|3ch | prediction RGB | ground-truth RGB].
We crop the *first* row only (sample 0).

Outputs:
  figures/cgan_phase2_first_sample_progression.png — input + epochs 1..50 + GT, numbered
  figures/cgan_phase2_first_sample_rgb_gray.png — ground-truth color beside grayscale input
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PREVIEW_DIR = Path("runs/my_cgan_study/phase2_best")
OUT_PROGRESSION = Path("figures/cgan_phase2_first_sample_progression.png")
OUT_PAIR = Path("figures/cgan_phase2_first_sample_rgb_gray.png")

EPOCHS = list(range(1, 51))
IMG = 128
PAD = 2
# First row of make_grid: sample 0 — [gray | pred | gt]
TOP_Y0, TOP_Y1 = PAD, PAD + IMG
GRAY_X = (PAD, PAD + IMG)
PRED_X = (PAD + IMG + PAD, PAD + IMG + PAD + IMG)
GT_X = (PAD + IMG + PAD + IMG + PAD, PAD + IMG + PAD + IMG + PAD + IMG)


def crop(im: Image.Image, x_range: tuple[int, int], y_range: tuple[int, int]) -> Image.Image:
    x0, x1 = x_range
    y0, y1 = y_range
    return im.crop((x0, y0, x1, y1))


def labeled_cell(
    im: Image.Image,
    label: str,
    cell_w: int,
    cell_h: int,
    font: ImageFont.FreeTypeFont,
    label_h: int,
) -> Image.Image:
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


def _default_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ):
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def build_progression() -> None:
    preview_paths = {e: PREVIEW_DIR / f"preview_epoch_{e:03d}.png" for e in EPOCHS}
    missing = [str(p) for p in preview_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preview PNGs ({len(missing)}): {missing[:5]}...")

    last_im = Image.open(preview_paths[EPOCHS[-1]]).convert("RGB")
    gray0 = crop(last_im, GRAY_X, (TOP_Y0, TOP_Y1))
    target_rgb = crop(last_im, GT_X, (TOP_Y0, TOP_Y1))

    predictions = []
    for e in EPOCHS:
        im = Image.open(preview_paths[e]).convert("RGB")
        predictions.append(crop(im, PRED_X, (TOP_Y0, TOP_Y1)))

    # 52 cells: Input + 50 epochs + GT in a 13 × 4 grid
    cells: list[tuple[str, Image.Image]] = [("Input (L)", gray0)]
    for e, pred in zip(EPOCHS, predictions):
        cells.append((f"Epoch {e}", pred))
    cells.append(("Ground truth", target_rgb))

    assert len(cells) == 52, f"expected 52 cells, got {len(cells)}"

    rows, cols = 4, 13
    cell_w = 120
    label_h = 30
    cell_img_h = cell_w  # square thumbs
    cell_h = cell_img_h + label_h
    margin = 12
    gutter = 6

    grid_w = cols * cell_w + (cols - 1) * gutter + 2 * margin
    grid_h = rows * cell_h + (rows - 1) * gutter + 2 * margin
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    font = _default_font(14)

    for idx, (label, im) in enumerate(cells):
        r = idx // cols
        c = idx % cols
        cell = labeled_cell(im, label, cell_w, cell_h, font, label_h)
        x = margin + c * (cell_w + gutter)
        y = margin + r * (cell_h + gutter)
        grid.paste(cell, (x, y))

    OUT_PROGRESSION.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT_PROGRESSION, "PNG", optimize=True)
    print(f"Saved {OUT_PROGRESSION} ({grid.size[0]}×{grid.size[1]})")


def build_rgb_gray_pair() -> None:
    preview = PREVIEW_DIR / f"preview_epoch_{EPOCHS[-1]:03d}.png"
    if not preview.exists():
        raise FileNotFoundError(preview)
    im = Image.open(preview).convert("RGB")
    gray0 = crop(im, GRAY_X, (TOP_Y0, TOP_Y1))
    rgb0 = crop(im, GT_X, (TOP_Y0, TOP_Y1))

    panel_w = 420
    label_h = 36
    img_h = panel_w
    cell_h = img_h + label_h
    gutter = 16
    margin = 20
    out_w = 2 * panel_w + gutter + 2 * margin
    out_h = cell_h + 2 * margin

    font = _default_font(20)
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    def paste_panel(x0: int, image: Image.Image, title: str) -> None:
        cell = labeled_cell(image, title, panel_w, cell_h, font, label_h)
        canvas.paste(cell, (x0, margin))

    paste_panel(margin, rgb0, "Original (RGB)")
    paste_panel(margin + panel_w + gutter, gray0, "Grayscale (L)")

    canvas.save(OUT_PAIR, "PNG", optimize=True)
    print(f"Saved {OUT_PAIR} ({canvas.size[0]}×{canvas.size[1]})")


def main() -> None:
    build_progression()
    build_rgb_gray_pair()


if __name__ == "__main__":
    main()
