"""Generate quantitative figures for the report.

Outputs (saved to ./figures/):
    controlnet_curves.png   - ControlNet train/val MSE + sample PSNR/SSIM vs epoch
    cgan_phase2_curves.png  - cGAN phase-2 best run: L1 (train/val) and GAN losses
    cgan_phase1_sweep.png   - cGAN phase-1 hyperparameter sweep best_val_l1 bar chart

Re-run any time the underlying CSVs change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
})


def plot_controlnet_curves() -> Path:
    df = pd.read_csv("runs/controlnet_softedge/epoch_metrics.csv")
    out = FIG_DIR / "controlnet_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_mse"], marker="o", label="train MSE", color="#1f77b4")
    ax.plot(df["epoch"], df["val_mse"], marker="s", label="val MSE", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Latent noise-prediction MSE")
    ax.set_title("(a) ControlNet training / validation MSE")
    ax.set_xticks(df["epoch"])
    ax.legend(loc="upper right")

    ax = axes[1]
    color_psnr = "#2ca02c"
    color_ssim = "#9467bd"
    l1, = ax.plot(df["epoch"], df["sample_psnr"], marker="o", color=color_psnr, label="PSNR (dB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)", color=color_psnr)
    ax.tick_params(axis="y", labelcolor=color_psnr)
    ax.set_xticks(df["epoch"])
    ax.set_title("(b) Sample PSNR / SSIM on fixed preview batch")

    ax2 = ax.twinx()
    ax2.grid(False)
    l2, = ax2.plot(df["epoch"], df["sample_ssim"], marker="s", color=color_ssim, label="SSIM")
    ax2.set_ylabel("SSIM", color=color_ssim)
    ax2.tick_params(axis="y", labelcolor=color_ssim)

    ax.legend(handles=[l1, l2], loc="lower right")

    fig.suptitle("ControlNet (softedge) learning curves over 10 epochs", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cgan_phase2_curves() -> Path:
    df = pd.read_csv("runs/my_cgan_study/phase2_best/epoch_metrics.csv")
    out = FIG_DIR / "cgan_phase2_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_l1"], label="train L1", color="#1f77b4")
    ax.plot(df["epoch"], df["val_l1"], label="val L1", color="#d62728")
    best_epoch = int(df["val_l1"].idxmin()) + 1
    best_val = float(df["val_l1"].min())
    ax.axvline(best_epoch, ls=":", color="#444", lw=1)
    ax.scatter([best_epoch], [best_val], color="#d62728", zorder=5)
    ax.annotate(
        f"best val L1 = {best_val:.4f}\n@ epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 2, best_val + 0.005),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"L1 on $ab$ channels")
    ax.set_title("(a) cGAN colorization loss")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(df["epoch"], df["train_g"], label="generator G (adv + L1)", color="#1f77b4")
    ax.plot(df["epoch"], df["train_d"], label="discriminator D", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("(b) cGAN adversarial dynamics")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.legend(loc="upper right")

    fig.suptitle(
        "cGAN phase-2 best configuration over 50 epochs (4,000-image subset, $128\\times128$)",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cgan_phase1_sweep() -> Path:
    df = pd.read_csv("runs/my_cgan_study/phase1/experiment_results.csv")
    out = FIG_DIR / "cgan_phase1_sweep.png"

    df = df.sort_values(["lr_g", "lr_d", "lambda_l1"]).reset_index(drop=True)

    lr_pairs = sorted({(r.lr_g, r.lr_d) for r in df.itertuples()})
    lambdas = sorted(df["lambda_l1"].unique())

    n_groups = len(lr_pairs)
    n_lambda = len(lambdas)
    width = 0.8 / n_lambda
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))

    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, lam in enumerate(lambdas):
        ys = []
        for lr_g, lr_d in lr_pairs:
            row = df[(df["lr_g"] == lr_g) & (df["lr_d"] == lr_d) & (df["lambda_l1"] == lam)]
            ys.append(float(row["best_val_l1"].iloc[0]) if len(row) else np.nan)
        offset = (i - (n_lambda - 1) / 2) * width
        bars = ax.bar(x + offset, ys, width=width * 0.95,
                      label=fr"$\lambda_{{L1}}={int(lam)}$",
                      color=palette[i % len(palette)], edgecolor="black", linewidth=0.4)
        for rect, val in zip(bars, ys):
            if np.isnan(val):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2.0, val,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    best_idx = df["best_val_l1"].idxmin()
    best_row = df.loc[best_idx]
    best_pair = (best_row.lr_g, best_row.lr_d)
    best_lam = best_row.lambda_l1
    gi = lr_pairs.index(best_pair)
    li = lambdas.index(best_lam)
    offset = (li - (n_lambda - 1) / 2) * width
    star_y = best_row.best_val_l1 - 0.008
    ax.scatter([gi + offset], [star_y], marker="*",
               color="gold", edgecolor="black", linewidths=0.7, s=240,
               zorder=10, label="phase-1 winner")

    ax.set_xticks(x)
    ax.set_xticklabels([f"lr_G={g:g}\nlr_D={d:g}" for g, d in lr_pairs], fontsize=9)
    ax.set_ylabel(r"Best val L1 on $ab$ (lower is better)")
    ax.set_xlabel("Learning-rate pair")
    ax.set_title("cGAN phase-1 hyperparameter sweep "
                 f"(10 epochs, 4,000 images, $128\\times128$, "
                 f"{len(df)} configurations)")
    ax.set_ylim(top=max(df["best_val_l1"]) * 1.22)
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.margins(x=0.02)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    paths = [
        plot_controlnet_curves(),
        plot_cgan_phase2_curves(),
        plot_cgan_phase1_sweep(),
    ]
    for p in paths:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
