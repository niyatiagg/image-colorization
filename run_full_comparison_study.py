#!/usr/bin/env python3
"""
Run baseline CNN, cGAN study, and ControlNet recolor, then aggregate stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_py(script: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    if proc.returncode != 0:
        raise SystemExit(f"Failed with exit code {proc.returncode}: {script.name}")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _best_row(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    return min(rows, key=lambda r: float(r[key]))


def _resolve_dir(path_like: str) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_stats_from_epoch_csv(
    epoch_csv: Path, train_col: str, val_col: str
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    rows = _read_csv(epoch_csv)
    if not rows:
        return None, None, None, None
    first_val = _safe_float(rows[0].get(val_col))
    best_val = None
    last_train = _safe_float(rows[-1].get(train_col))
    last_val = _safe_float(rows[-1].get(val_col))
    vals = [_safe_float(r.get(val_col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if vals:
        best_val = min(vals)
    return first_val, best_val, last_train, last_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified experiment runner for report-ready stats.")
    p.add_argument("--study-root", type=str, default="./runs/full_comparison_study")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--subset-size", type=int, default=4000)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)

    p.add_argument("--baseline-epochs", type=int, default=10)
    p.add_argument("--baseline-lr", type=str, default="1e-3")
    p.add_argument("--baseline-batch-size", type=str, default="32")

    p.add_argument("--cgan-phase1-epochs", type=int, default=10)
    p.add_argument("--cgan-phase2-epochs", type=int, default=50)
    p.add_argument("--cgan-batch-size", type=int, default=32)
    p.add_argument("--cgan-lr-g", type=str, default="1e-4,2e-4")
    p.add_argument("--cgan-lr-d", type=str, default="1e-4,2e-4")
    p.add_argument("--cgan-lambda-l1", type=str, default="50,100,200")

    p.add_argument("--controlnet-epochs", type=int, default=10)
    p.add_argument("--controlnet-lr", type=float, default=1e-5)
    p.add_argument("--controlnet-batch-size", type=int, default=4)
    p.add_argument("--controlnet-weight-decay", type=float, default=1e-2)
    p.add_argument("--controlnet-base-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet-amp", action="store_true")

    p.add_argument("--skip-baseline", action="store_true", help="Do not run baseline; reuse existing results.")
    p.add_argument("--skip-cgan", action="store_true", help="Do not run cGAN; reuse existing results.")
    p.add_argument("--skip-controlnet", action="store_true", help="Do not run ControlNet; reuse existing results.")

    p.add_argument(
        "--existing-baseline-dir",
        type=str,
        default="",
        help="Existing baseline run directory containing experiment_results.csv and optional epoch_metrics.csv files.",
    )
    p.add_argument(
        "--existing-cgan-study-dir",
        type=str,
        default="",
        help="Existing cGAN study directory containing phase1/ and phase2_best/.",
    )
    p.add_argument(
        "--existing-controlnet-dir",
        type=str,
        default="",
        help="Existing ControlNet run directory containing epoch_metrics.csv.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.study_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    baseline_dir = (
        _resolve_dir(args.existing_baseline_dir)
        if args.existing_baseline_dir
        else (root / "baseline")
    )
    cgan_dir = (
        _resolve_dir(args.existing_cgan_study_dir)
        if args.existing_cgan_study_dir
        else (root / "cgan")
    )
    controlnet_dir = (
        _resolve_dir(args.existing_controlnet_dir)
        if args.existing_controlnet_dir
        else (root / "controlnet")
    )

    # 1) Baseline CNN sweep
    if not args.skip_baseline:
        _run_py(
            _repo_root() / "baseline_cnn_places365.py",
            [
                "--data-root",
                args.data_root,
                "--output-dir",
                str(baseline_dir),
                "--subset-size",
                str(args.subset_size),
                "--image-size",
                str(args.image_size),
                "--val-fraction",
                str(args.val_fraction),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.baseline_epochs),
                "--lr",
                *[x.strip() for x in args.baseline_lr.split(",") if x.strip()],
                "--batch-size",
                *[x.strip() for x in args.baseline_batch_size.split(",") if x.strip()],
            ],
        )

    # 2) cGAN 2-phase study
    if not args.skip_cgan:
        _run_py(
            _repo_root() / "run_cgan_hyperparameter_study.py",
            [
                "--study-root",
                str(cgan_dir),
                "--data-root",
                args.data_root,
                "--subset-size",
                str(args.subset_size),
                "--image-size",
                str(args.image_size),
                "--batch-size",
                str(args.cgan_batch_size),
                "--val-fraction",
                str(args.val_fraction),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--phase1-epochs",
                str(args.cgan_phase1_epochs),
                "--phase2-epochs",
                str(args.cgan_phase2_epochs),
                "--lr-g",
                args.cgan_lr_g,
                "--lr-d",
                args.cgan_lr_d,
                "--lambda-l1",
                args.cgan_lambda_l1,
            ],
        )

    # 3) ControlNet single run
    controlnet_cmd = [
        "--data-root",
        args.data_root,
        "--output-dir",
        str(controlnet_dir),
        "--subset-size",
        str(args.subset_size),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.controlnet_batch_size),
        "--epochs",
        str(args.controlnet_epochs),
        "--lr",
        str(args.controlnet_lr),
        "--weight-decay",
        str(args.controlnet_weight_decay),
        "--val-fraction",
        str(args.val_fraction),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--base-model-id",
        args.controlnet_base_model_id,
    ]
    if args.controlnet_amp:
        controlnet_cmd.append("--amp")
    if not args.skip_controlnet:
        _run_py(_repo_root() / "controlnet_recolor_places365.py", controlnet_cmd)

    # Aggregate summary
    baseline_rows = _read_csv(baseline_dir / "experiment_results.csv")
    cgan_phase1_rows = _read_csv(cgan_dir / "phase1" / "experiment_results.csv")
    cgan_phase2_rows = _read_csv(cgan_dir / "phase2_best" / "experiment_results.csv")
    controlnet_rows = [
        {
            "output_dir": str(controlnet_dir),
            "epoch_metrics_csv": str(controlnet_dir / "epoch_metrics.csv"),
        }
    ]

    summary_rows: List[Dict[str, Any]] = []
    b_best = _best_row(baseline_rows, "best_val_l1")
    if b_best:
        b_epoch_csv = _resolve_dir(b_best["output_dir"]) / "epoch_metrics.csv"
        b_first, b_best_from_curve, b_last_train, b_last_val = _metric_stats_from_epoch_csv(
            b_epoch_csv, "train_l1", "val_l1"
        )
        b_best_val = _safe_float(b_best.get("best_val_l1"))
        if b_best_val is None:
            b_best_val = b_best_from_curve
        if b_last_train is None:
            b_last_train = _safe_float(b_best.get("final_train_l1"))
        if b_last_val is None:
            b_last_val = _safe_float(b_best.get("final_val_l1"))
        b_rel = None
        if b_first is not None and b_best_val is not None and b_first > 0:
            b_rel = (b_first - b_best_val) / b_first
        summary_rows.append(
            {
                "model": "baseline_cnn",
                "selection": "best_over_grid_by_best_val_l1",
                "best_val_metric": b_best_val if b_best_val is not None else "",
                "final_train_metric": b_last_train if b_last_train is not None else "",
                "final_val_metric": b_last_val if b_last_val is not None else "",
                "primary_metric_name": "l1",
                "output_dir": b_best["output_dir"],
                "epoch_metrics_csv": str(b_epoch_csv),
                "best_epoch_relative_improvement": b_rel if b_rel is not None else "",
            }
        )

    c1_best = _best_row(cgan_phase1_rows, "best_val_l1")
    if c1_best:
        c1_epoch_csv = _resolve_dir(c1_best["output_dir"]) / "epoch_metrics.csv"
        c1_first, c1_best_from_curve, c1_last_train, c1_last_val = _metric_stats_from_epoch_csv(
            c1_epoch_csv, "train_l1", "val_l1"
        )
        c1_best_val = _safe_float(c1_best.get("best_val_l1"))
        if c1_best_val is None:
            c1_best_val = c1_best_from_curve
        if c1_last_train is None:
            c1_last_train = _safe_float(c1_best.get("final_train_l1"))
        if c1_last_val is None:
            c1_last_val = _safe_float(c1_best.get("final_val_l1"))
        c1_rel = None
        if c1_first is not None and c1_best_val is not None and c1_first > 0:
            c1_rel = (c1_first - c1_best_val) / c1_first
        summary_rows.append(
            {
                "model": "cgan_phase1",
                "selection": "best_over_grid_by_best_val_l1",
                "best_val_metric": c1_best_val if c1_best_val is not None else "",
                "final_train_metric": c1_last_train if c1_last_train is not None else "",
                "final_val_metric": c1_last_val if c1_last_val is not None else "",
                "primary_metric_name": "l1",
                "output_dir": c1_best["output_dir"],
                "epoch_metrics_csv": str(c1_epoch_csv),
                "best_epoch_relative_improvement": c1_rel if c1_rel is not None else "",
            }
        )

    if cgan_phase2_rows:
        c2 = cgan_phase2_rows[0]
        c2_epoch_csv = _resolve_dir(c2["output_dir"]) / "epoch_metrics.csv"
        c2_first, c2_best_from_curve, c2_last_train, c2_last_val = _metric_stats_from_epoch_csv(
            c2_epoch_csv, "train_l1", "val_l1"
        )
        c2_best_val = _safe_float(c2.get("best_val_l1"))
        if c2_best_val is None:
            c2_best_val = c2_best_from_curve
        if c2_last_train is None:
            c2_last_train = _safe_float(c2.get("final_train_l1"))
        if c2_last_val is None:
            c2_last_val = _safe_float(c2.get("final_val_l1"))
        c2_rel = None
        if c2_first is not None and c2_best_val is not None and c2_first > 0:
            c2_rel = (c2_first - c2_best_val) / c2_first
        summary_rows.append(
            {
                "model": "cgan_phase2_best",
                "selection": "retrain_best_phase1_config",
                "best_val_metric": c2_best_val if c2_best_val is not None else "",
                "final_train_metric": c2_last_train if c2_last_train is not None else "",
                "final_val_metric": c2_last_val if c2_last_val is not None else "",
                "primary_metric_name": "l1",
                "output_dir": c2["output_dir"],
                "epoch_metrics_csv": str(c2_epoch_csv),
                "best_epoch_relative_improvement": c2_rel if c2_rel is not None else "",
            }
        )

    c_metrics = _read_csv(controlnet_dir / "epoch_metrics.csv")
    if c_metrics:
        best = min(c_metrics, key=lambda r: float(r["val_mse"]))
        last = c_metrics[-1]
        first_val = _safe_float(c_metrics[0].get("val_mse"))
        best_val = _safe_float(best.get("val_mse"))
        rel = None
        if first_val is not None and best_val is not None and first_val > 0:
            rel = (first_val - best_val) / first_val
        summary_rows.append(
            {
                "model": "controlnet_recolor",
                "selection": "single_run_best_epoch_by_val_mse",
                "best_val_metric": best["val_mse"],
                "final_train_metric": last["train_mse"],
                "final_val_metric": last["val_mse"],
                "primary_metric_name": "mse",
                "output_dir": str(controlnet_dir),
                "epoch_metrics_csv": str(controlnet_dir / "epoch_metrics.csv"),
                "best_epoch_relative_improvement": rel if rel is not None else "",
            }
        )

    summary_csv = root / "comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fields = [
            "model",
            "selection",
            "primary_metric_name",
            "best_val_metric",
            "final_train_metric",
            "final_val_metric",
            "best_epoch_relative_improvement",
            "output_dir",
            "epoch_metrics_csv",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    report_json = root / "comparison_report.json"
    payload = {
        "study_root": str(root),
        "baseline_rows": baseline_rows,
        "cgan_phase1_rows": cgan_phase1_rows,
        "cgan_phase2_rows": cgan_phase2_rows,
        "controlnet_rows": controlnet_rows,
        "comparison_summary_csv": str(summary_csv),
    }
    with open(report_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_json}")


if __name__ == "__main__":
    main()
