#!/usr/bin/env python3
"""Rebuild per-epoch CSVs for a CNN baseline run when epoch_metrics.csv is missing.

The training script normally writes ``epoch_metrics.csv`` (epoch, train_l1, val_l1).
This tool reconstructs *val_l1* for every epoch from ``preview_epoch_*.png`` by
cropping the first validation minibatch (4×[gray|pred|gt]) and computing mean L1 on
normalized *ab* channels (same definition as training), after round-tripping through
the saved RGB preview — so values are close to, but not identical to, the original log.

*train_l1* cannot be recovered from previews. For the last epoch row only, this script
optionally runs a forward-only pass over train/val loaders with ``last_model.pt`` and
fills *train_l1* and overwrites *val_l1* with the exact full-split metrics (requires
Places365 under ``--data-root``).

Writes:
  - ``--output-dir/epoch_results.csv`` (same rows as below)
  - ``--output-dir/epoch_metrics.csv`` (alias for tools expecting that name)

If ``experiment_results.csv`` is absent, creates a one-row summary using full-split
eval on both ``best_model.pt`` and ``last_model.pt`` (whichever achieves lower val L1).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMG = 128
PAD = 2


def _crop_triplet_row(im: Image.Image, row_index: int) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """One row of make_grid(..., nrow=3): [gray | pred | gt]. row_index in 0..3."""
    y0 = PAD + row_index * (IMG + PAD)
    y1 = y0 + IMG
    gray_x = (PAD, PAD + IMG)
    pred_x = (PAD + IMG + PAD, PAD + IMG + PAD + IMG)
    gt_x = (PAD + IMG + PAD + IMG + PAD, PAD + IMG + PAD + IMG + PAD + IMG)
    return (
        im.crop((gray_x[0], y0, gray_x[1], y1)),
        im.crop((pred_x[0], y0, pred_x[1], y1)),
        im.crop((gt_x[0], y0, gt_x[1], y1)),
    )


def _rgb_to_ab_norm(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 [H,W,3] -> ab [H,W,2] normalized like PlacesColorizationDataset."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return (lab[:, :, 1:3] - 128.0) / 128.0


def preview_val_ab_l1(path: Path) -> float:
    im = Image.open(path).convert("RGB")
    losses: List[float] = []
    for r in range(4):
        _, pred, gt = _crop_triplet_row(im, r)
        p_ab = _rgb_to_ab_norm(np.asarray(pred))
        g_ab = _rgb_to_ab_norm(np.asarray(gt))
        losses.append(float(np.mean(np.abs(p_ab - g_ab))))
    return float(np.mean(losses))


def _discover_epochs(out_dir: Path) -> List[int]:
    epochs: List[int] = []
    for p in out_dir.glob("preview_epoch_*.png"):
        stem = p.stem  # preview_epoch_NNN
        try:
            epochs.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(epochs)


def _full_split_metrics(
    data_root: str,
    subset_size: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    ckpt_path: Path,
    device: str,
) -> Tuple[float, float]:
    import torch
    import torch.nn as nn
    from baseline_cnn_places365 import ColorizationCNN, build_dataloaders, set_seed

    set_seed(seed)
    dev = torch.device(device)
    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        subset_size=subset_size,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=val_fraction,
        seed=seed,
    )
    model = ColorizationCNN().to(dev)
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    crit = nn.L1Loss()

    def split_loss(loader) -> float:
        total, n = 0.0, 0
        with torch.no_grad():
            for l_in, ab_t in loader:
                l_in = l_in.to(dev)
                ab_t = ab_t.to(dev)
                pred = model(l_in)
                total += crit(pred, ab_t).item() * l_in.size(0)
                n += l_in.size(0)
        return total / max(1, n)

    return split_loss(train_loader), split_loss(val_loader)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=str, default="./runs/cnn_baseline")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--subset-size", type=int, default=4000)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-full-eval", action="store_true", help="Do not run torch eval; val from previews only.")
    p.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.output_dir)
    epochs = _discover_epochs(out)
    if not epochs:
        raise SystemExit(f"No preview_epoch_*.png under {out}")

    rows: List[dict] = []
    for ep in epochs:
        prev = out / f"preview_epoch_{ep:03d}.png"
        if not prev.is_file():
            raise FileNotFoundError(prev)
        rows.append(
            {
                "epoch": str(ep),
                "train_l1": "",
                "val_l1": f"{preview_val_ab_l1(prev):.8f}",
            }
        )

    last_ep = max(epochs)
    train_full: Optional[float] = None
    val_full_last: Optional[float] = None
    val_full_best: Optional[float] = None

    if not args.skip_full_eval:
        try:
            import torch

            last_pt = out / "last_model.pt"
            best_pt = out / "best_model.pt"
            if last_pt.is_file():
                train_full, val_full_last = _full_split_metrics(
                    args.data_root,
                    args.subset_size,
                    args.image_size,
                    args.batch_size,
                    args.num_workers,
                    args.val_fraction,
                    args.seed,
                    last_pt,
                    args.device,
                )
            if best_pt.is_file():
                _, val_full_best = _full_split_metrics(
                    args.data_root,
                    args.subset_size,
                    args.image_size,
                    args.batch_size,
                    args.num_workers,
                    args.val_fraction,
                    args.seed,
                    best_pt,
                    args.device,
                )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: full-split eval skipped ({exc})")

    if train_full is not None and val_full_last is not None:
        for r in rows:
            if int(r["epoch"]) == last_ep:
                r["train_l1"] = f"{train_full:.8f}"
                r["val_l1"] = f"{val_full_last:.8f}"
                break

    fieldnames = ["epoch", "train_l1", "val_l1"]
    for name in ("epoch_metrics.csv", "epoch_results.csv"):
        dest = out / name
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {dest} ({len(rows)} rows)")

    exp_path = out / "experiment_results.csv"
    if not exp_path.is_file() and train_full is not None and val_full_last is not None:
        best_val = val_full_last
        if val_full_best is not None:
            best_val = min(best_val, val_full_best)
        row = {
            "lr": str(1e-3),
            "batch_size": str(args.batch_size),
            "best_val_l1": f"{best_val:.8f}",
            "final_train_l1": f"{train_full:.8f}",
            "final_val_l1": f"{val_full_last:.8f}",
            "output_dir": str(out.resolve()),
            "epochs": str(last_ep),
            "subset_size": str(args.subset_size),
            "seed": str(args.seed),
        }
        with open(exp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Wrote {exp_path} (reconstructed summary; best_val_l1=min(last,best) val L1)")
    elif exp_path.is_file():
        print(f"Leaving existing {exp_path}")


if __name__ == "__main__":
    main()
