#!/usr/bin/env python3
"""Draw colorization pipeline flowcharts as separate vector figures (no overlapping edges).

Writes by default:
  figures/pipeline_path_a_cnn.{png,pdf}       — Path A CNN baseline
  figures/pipeline_path_b_cgan.{png,pdf}      — Path B conditional GAN + train loop
  figures/pipeline_path_c_controlnet.{png,pdf} — Path C ControlNet + luminance branch

Optional:
  python build_pipeline_flowchart.py --classic
    also writes figures/pipeline_three_paths_classic.{png,pdf} (combined legacy layout).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = Path("figures")
DPI = 150


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    *,
    dashed_edge: bool = False,
    fontsize: int = 9,
    z: int = 2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=face,
            linestyle="--" if dashed_edge else "-",
            zorder=z,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        linespacing=1.12,
        zorder=z + 1,
    )


def _arrow(
    ax,
    xy1,
    xy2,
    *,
    dashed: bool = False,
    rad: float = 0.0,
    z: float = 1.5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="Simple, tail_width=0.35, head_width=5, head_length=6",
            color="#222222",
            linewidth=1.0,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def _polyarrow(
    ax,
    pts: list[tuple[float, float]],
    *,
    dashed: bool = False,
    z: float = 1.2,
) -> None:
    p = np.asarray(pts, dtype=float)
    ax.plot(
        p[:, 0],
        p[:, 1],
        color="#222222",
        linewidth=1.0,
        linestyle="--" if dashed else "-",
        solid_joinstyle="round",
        zorder=z,
    )
    if len(p) < 2:
        return
    tail, tip = p[-2], p[-1]
    dv = tip - tail
    n = float(np.linalg.norm(dv))
    if n < 1e-9:
        return
    u = dv / n
    shrink = 6.0
    start = tip - shrink * u
    ax.add_patch(
        FancyArrowPatch(
            start,
            tip,
            arrowstyle="Simple, tail_width=0.35, head_width=5, head_length=6",
            color="#222222",
            linewidth=1.0,
            linestyle="--" if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=z + 0.05,
        )
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = stem.strip().replace("/", "").replace("\\", "") or "flowchart"
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white", edgecolor="none", dpi=DPI)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved", png, "and", pdf)


def draw_path_a() -> None:
    green, purple = "#c5e1c8", "#d8cee8"
    fig, ax = plt.subplots(figsize=(11.0, 2.8), dpi=DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_aspect("equal")

    y, h = 1.15, 0.72
    xg, wg = 0.45, 1.35
    _box(ax, xg, y, wg, h, r"Grayscale $L\in[0,1]$", green, fontsize=10)

    x0, w0 = 2.15, 1.55
    _box(ax, x0, y, w0, h, r"CNN  $L\rightarrow ab$", purple)
    x1, w1 = 3.95, 1.85
    _box(ax, x1, y, w1, h, r"recombine  $L+ab\rightarrow$ RGB", purple)
    x2, w2 = 6.15, 0.95
    _box(ax, x2, y, w2, h, "RGB", green)

    cy = y + h / 2
    _arrow(ax, (xg + wg, cy), (x0, cy))
    _arrow(ax, (x0 + w0, cy), (x1, cy))
    _arrow(ax, (x1 + w1, cy), (x2, cy))

    ax.text(0.45, y + h + 0.22, "Path A — CNN baseline", fontsize=11, color="#333333", style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12)
    _save(fig, "pipeline_path_a_cnn")


def draw_path_b() -> None:
    """Linear: gray → G → recombine → RGB → D; train signal D → G routed below row."""
    green, purple = "#c5e1c8", "#d8cee8"
    fig, ax = plt.subplots(figsize=(13.5, 4.2), dpi=DPI)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_aspect("equal")

    y, h = 2.35, 0.72
    xg, wg = 0.4, 1.35
    _box(ax, xg, y, wg, h, r"Grayscale $L\in[0,1]$", green, fontsize=10)

    w_g, w_r, w_i, w_d = 1.55, 1.85, 0.95, 1.85
    x_g = 2.05
    x_r = x_g + w_g + 0.35
    x_i = x_r + w_r + 0.35
    x_d = x_i + w_i + 0.35

    _box(ax, x_g, y, w_g, h, r"Generator $G$\n$[L;z]\rightarrow ab$", purple, fontsize=8)
    _box(ax, x_r, y, w_r, h, r"recombine  $L+ab\rightarrow$ RGB", purple)
    _box(ax, x_i, y, w_i, h, "RGB", green)
    _box(ax, x_d, y, w_d, h, r"PatchGAN $D$\n$[L,ab]\rightarrow$ real/fake", purple, fontsize=8)

    cy = y + h / 2
    _arrow(ax, (xg + wg, cy), (x_g, cy))
    _arrow(ax, (x_g + w_g, cy), (x_r, cy))
    _arrow(ax, (x_r + w_r, cy), (x_i, cy))
    _arrow(ax, (x_i + w_i, cy), (x_d, cy))

    # Train signal: PatchGAN bottom → under row → up into Generator (no box crossings)
    y_low = y - 0.95
    gx_mid = x_g + w_g * 0.5
    dx_mid = x_d + w_d * 0.5
    _polyarrow(
        ax,
        [
            (dx_mid, y),
            (dx_mid, y_low),
            (gx_mid, y_low),
            (gx_mid, y),
        ],
        dashed=True,
    )
    ax.text(
        gx_mid + 0.1,
        y_low - 0.18,
        "(train signal)",
        fontsize=8,
        ha="left",
        va="top",
        color="#555555",
        style="italic",
        zorder=3,
    )

    ax.text(0.4, y + h + 0.22, "Path B — Conditional GAN", fontsize=11, color="#333333", style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)
    _save(fig, "pipeline_path_b_cgan")


def draw_path_c() -> None:
    green, purple, pf = "#c5e1c8", "#d8cee8", "#e6e0ef"
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=DPI)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.set_aspect("equal")

    y_m, h = 2.55, 0.68
    xg, wg = 0.35, 1.3
    _box(ax, xg, y_m, wg, h, r"Grayscale $L\in[0,1]$", green, fontsize=10)

    xs = [2.05, 3.45, 4.95, 6.45, 7.95, 10.05]
    ws = [1.25, 1.35, 1.35, 1.35, 1.25, 1.15]
    labels = [
        "HED\nsoftedge",
        "SD-1.5 VAE\n(frozen)",
        "ControlNet\n(trained)",
        "SD-1.5 UNet\n(frozen)",
        "RGB",
        "RGB\nfinal",
    ]
    frozen = [False, True, False, True, False, False]

    for i, (xi, wi, lab, fr) in enumerate(zip(xs, ws, labels, frozen)):
        fc = pf if fr else purple
        _box(ax, xi, y_m, wi, h, lab, fc, dashed_edge=fr, fontsize=8)

    cy = y_m + h / 2
    _arrow(ax, (xg + wg, cy), (xs[0], cy))
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + ws[i], cy), (xs[i + 1], cy))

    # Luminance branch
    lx, ly, lw, lh = 0.35, 0.55, 2.85, 0.62
    _box(ax, lx, ly, lw, lh, "Luminance lock\n(Lab swap)", purple, fontsize=8)
    _arrow(ax, (xg + wg / 2, y_m), (lx + lw / 2, ly + lh), dashed=True)

    # Post-process: stay below main chain, then up into RGB final
    y_duct = y_m - 0.55
    cx_final = xs[-1] + ws[-1] / 2
    x_turn = 12.4
    lum_y = ly + lh * 0.55
    _polyarrow(
        ax,
        [
            (lx + lw, lum_y),
            (x_turn, lum_y),
            (x_turn, y_duct),
            (cx_final, y_duct),
            (cx_final, y_m + 0.06),
        ],
        dashed=False,
    )
    ax.text(
        x_turn + 0.15,
        (lum_y + y_duct) / 2,
        "post-process",
        fontsize=8,
        ha="left",
        va="center",
        color="#555555",
        style="italic",
        rotation=90,
        zorder=3,
    )

    ax.text(0.35, y_m + h + 0.22, "Path C — Edge-conditioned ControlNet", fontsize=11, color="#333333", style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    _save(fig, "pipeline_path_c_controlnet")


def draw_classic_combined() -> None:
    """Legacy single-canvas figure (PatchGAN to the side, simple arcs)."""
    green, purple, purple_frozen = "#c5e1c8", "#d8cee8", "#e6e0ef"
    fig, ax = plt.subplots(figsize=(14.0, 9.0), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_aspect("equal")

    gw, gh = 1.35, 0.95
    gx, gy = 0.35, 3.85
    _box(ax, gx, gy, gw, gh, r"Grayscale $L\in[0,1]$", green, fontsize=10)
    gcy = gy + gh / 2

    ya, w1, h1 = 6.55, 1.55, 0.72
    x0 = 2.15
    _box(ax, x0, ya, w1, h1, r"CNN  $L\rightarrow ab$", purple)
    _arrow(ax, (gx + gw, gcy), (x0, ya + h1 / 2))
    x1 = x0 + w1 + 0.35
    _box(ax, x1, ya, 1.85, h1, r"recombine  $L+ab\rightarrow$ RGB", purple)
    _arrow(ax, (x0 + w1, ya + h1 / 2), (x1, ya + h1 / 2))
    x2 = x1 + 1.85 + 0.35
    _box(ax, x2, ya, 0.95, h1, "RGB", green)
    _arrow(ax, (x1 + 1.85, ya + h1 / 2), (x2, ya + h1 / 2))
    ax.text(0.35, ya + h1 + 0.35, "Path A — CNN baseline", fontsize=10, color="#333333", style="italic")

    yb = 4.55
    _box(ax, x0, yb, w1, h1, r"Generator $G$\n$[L;z]\rightarrow ab$", purple, fontsize=8)
    _arrow(ax, (gx + gw, gcy), (x0, yb + h1 / 2), rad=0.12)
    _box(ax, x1, yb, 1.85, h1, r"recombine  $L+ab\rightarrow$ RGB", purple)
    _arrow(ax, (x0 + w1, yb + h1 / 2), (x1, yb + h1 / 2))
    _box(ax, x2, yb, 0.95, h1, "RGB", green)
    _arrow(ax, (x1 + 1.85, yb + h1 / 2), (x2, yb + h1 / 2))
    ax.text(0.35, yb + h1 + 0.35, "Path B — Conditional GAN", fontsize=10, color="#333333", style="italic")

    dw, dh = 1.75, 0.68
    dx, dy = x2 + 1.15, yb + 0.02
    _box(ax, dx, dy, dw, dh, r"PatchGAN $D$\n$[L,ab]\rightarrow$ real/fake", purple, fontsize=8)
    _arrow(ax, (x0 + w1 / 2, yb), (dx + dw / 2, dy + dh), dashed=True, rad=-0.15)
    ax.text(dx + dw / 2, dy + dh + 0.28, "train signal", fontsize=7, ha="center", color="#555555", style="italic")

    yc, h2 = 1.55, 0.68
    xs = [2.0, 3.45, 4.95, 6.45, 7.95, 10.05]
    ws = [1.25, 1.35, 1.35, 1.35, 1.25, 1.15]
    labels = [
        "HED\nsoftedge",
        "SD-1.5 VAE\n(frozen)",
        "ControlNet\n(trained)",
        "SD-1.5 UNet\n(frozen)",
        "RGB",
        "RGB\nfinal",
    ]
    frozen_edge = [False, True, False, True, False, False]
    _arrow(ax, (gx + gw, gcy), (xs[0], yc + h2 / 2), rad=-0.18)
    for i in range(len(xs)):
        fc = purple_frozen if frozen_edge[i] else purple
        _box(
            ax,
            xs[i],
            yc,
            ws[i],
            h2,
            labels[i],
            fc,
            dashed_edge=frozen_edge[i],
            fontsize=8,
        )
        if i > 0:
            _arrow(ax, (xs[i - 1] + ws[i - 1], yc + h2 / 2), (xs[i], yc + h2 / 2))
    ax.text(0.35, yc + h2 + 0.35, "Path C — Edge-conditioned ControlNet", fontsize=10, color="#333333", style="italic")

    lx, ly, lw, lh = 0.35, 0.28, 3.0, 0.58
    _box(ax, lx, ly, lw, lh, "Luminance lock\n(Lab swap)", purple, fontsize=8)
    _arrow(ax, (gx + gw / 2, gy), (lx + lw / 2, ly + lh), dashed=True)
    rgb_mid_y = yc + h2 / 2
    _arrow(ax, (lx + lw, ly + lh / 2), (xs[-1], rgb_mid_y), rad=0.08)
    ax.text(7.0, 0.92, "post-process", fontsize=7, ha="center", color="#555555", style="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    _save(fig, "pipeline_three_paths_classic")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--classic",
        action="store_true",
        help="Also emit the combined legacy figure pipeline_three_paths_classic.",
    )
    args = p.parse_args()

    draw_path_a()
    draw_path_b()
    draw_path_c()
    if args.classic:
        draw_classic_combined()


if __name__ == "__main__":
    main()
