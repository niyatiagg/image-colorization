#!/usr/bin/env python3
"""
Two-phase CGAN hyperparameter study.

Phase 1: runs ``cgan_places365.py`` once with the full Cartesian product of
``--lr-g``, ``--lr-d``, and ``--lambda-l1`` for ``--phase1-epochs`` (default 10).

Phase 2: retrains the single best configuration (lowest ``best_val_l1`` from
phase 1) for ``--phase2-epochs`` (default 50).

Writes a JSON report and a flat CSV for downstream report writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_cgan(
    cgan_script: Path,
    argv_extra: List[str],
) -> None:
    cmd = [sys.executable, str(cgan_script), *argv_extra]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    if proc.returncode != 0:
        raise SystemExit(
            f"cgan_places365.py failed with exit code {proc.returncode}"
        )


def _read_experiment_results_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pick_best_phase1_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    def key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        b = float(r["best_val_l1"])
        lg = float(r["lr_g"])
        ld = float(r["lr_d"])
        l1 = float(r["lambda_l1"])
        return (b, lg, ld, l1)

    return min(rows, key=key)


def _float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-1 grid (10 ep) + phase-2 best run (50 ep) for CGAN."
    )
    p.add_argument(
        "--study-root",
        type=str,
        default="./runs/cgan_hyperparameter_study",
        help="Root directory for phase1/, phase2_best/, and report files.",
    )
    p.add_argument(
        "--cgan-script",
        type=str,
        default=str(_repo_root() / "cgan_places365.py"),
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--subset-size", type=int, default=4000)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase2-epochs", type=int, default=50)
    p.add_argument(
        "--lr-g",
        type=str,
        default="1e-4,2e-4",
        help="Comma-separated generator LRs (full grid with lr-d and lambda-l1).",
    )
    p.add_argument(
        "--lr-d",
        type=str,
        default="1e-4,2e-4",
        help="Comma-separated discriminator LRs.",
    )
    p.add_argument(
        "--lambda-l1",
        type=str,
        default="50,100,200",
        help="Comma-separated L1 weights on ab.",
    )
    p.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Defaults to {study-root}/study_report.json",
    )
    p.add_argument(
        "--report-csv",
        type=str,
        default="",
        help="Defaults to {study-root}/study_report.csv",
    )
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Reuse existing phase1/experiment_results.csv (must exist).",
    )
    p.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Only run phase 1 and write report (no 50-epoch fine run).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.study_root).resolve()
    phase1_dir = root / "phase1"
    phase2_dir = root / "phase2_best"
    root.mkdir(parents=True, exist_ok=True)

    cgan = Path(args.cgan_script).resolve()
    if not cgan.is_file():
        raise SystemExit(f"CGAN script not found: {cgan}")

    lr_g = _float_list(args.lr_g)
    lr_d = _float_list(args.lr_d)
    lambda_l1 = _float_list(args.lambda_l1)
    n_combo = len(lr_g) * len(lr_d) * len(lambda_l1)
    print(
        f"Grid size: {len(lr_g)} x {len(lr_d)} x {len(lambda_l1)} = {n_combo} runs",
        flush=True,
    )

    report_json = Path(args.report_json) if args.report_json else root / "study_report.json"
    report_csv = Path(args.report_csv) if args.report_csv else root / "study_report.csv"

    common = [
        "--data-root",
        args.data_root,
        "--subset-size",
        str(args.subset_size),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--val-fraction",
        str(args.val_fraction),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]

    if not args.skip_phase1:
        phase1_dir.mkdir(parents=True, exist_ok=True)
        p1 = [
            "--output-dir",
            str(phase1_dir),
            "--epochs",
            str(args.phase1_epochs),
            "--lr-g",
            *[str(x) for x in lr_g],
            "--lr-d",
            *[str(x) for x in lr_d],
            "--lambda-l1",
            *[str(x) for x in lambda_l1],
        ]
        _run_cgan(cgan, common + p1)

    exp1_path = phase1_dir / "experiment_results.csv"
    phase1_rows = _read_experiment_results_csv(exp1_path)
    if not phase1_rows:
        raise SystemExit(f"No rows in {exp1_path}; phase 1 did not complete.")

    best = _pick_best_phase1_row(phase1_rows)
    assert best is not None
    best_lr_g = float(best["lr_g"])
    best_lr_d = float(best["lr_d"])
    best_l1 = float(best["lambda_l1"])

    print(
        f"Best phase-1 (min best_val_l1): lr_g={best_lr_g} lr_d={best_lr_d} "
        f"lambda_l1={best_l1} best_val_l1={best['best_val_l1']}",
        flush=True,
    )

    exp2_path = phase2_dir / "experiment_results.csv"
    phase2_row: Optional[Dict[str, Any]] = None
    if not args.skip_phase2:
        phase2_dir.mkdir(parents=True, exist_ok=True)
        p2 = [
            "--output-dir",
            str(phase2_dir),
            "--epochs",
            str(args.phase2_epochs),
            "--lr-g",
            str(best_lr_g),
            "--lr-d",
            str(best_lr_d),
            "--lambda-l1",
            str(best_l1),
        ]
        _run_cgan(cgan, common + p2)
        rows2 = _read_experiment_results_csv(exp2_path)
        phase2_row = rows2[0] if rows2 else None
    elif exp2_path.is_file():
        rows2 = _read_experiment_results_csv(exp2_path)
        phase2_row = rows2[0] if rows2 else None

    report: Dict[str, Any] = {
        "selection_metric": "best_val_l1 (minimum over training; checkpoint when val L1 improved)",
        "phase1": {
            "epochs": args.phase1_epochs,
            "output_dir": str(phase1_dir),
            "experiment_results_csv": str(exp1_path),
            "combinations": phase1_rows,
        },
        "best_from_phase1": best,
        "phase2": None
        if phase2_row is None
        else {
            "epochs": args.phase2_epochs,
            "output_dir": str(phase2_dir),
            "experiment_results_csv": str(exp2_path),
            "row": phase2_row,
        },
    }

    combos_out: List[Dict[str, Any]] = []
    for r in phase1_rows:
        od = Path(r["output_dir"])
        if not od.is_absolute():
            od = (_repo_root() / od).resolve()
        combos_out.append(
            {
                **r,
                "epoch_metrics_csv": str(od / "epoch_metrics.csv"),
            }
        )
    report["phase1"]["combinations"] = combos_out
    if report["phase2"] is not None and phase2_row is not None:
        od2 = Path(phase2_row["output_dir"])
        if not od2.is_absolute():
            od2 = (_repo_root() / od2).resolve()
        report["phase2"] = {
            **report["phase2"],
            "row": {
                **phase2_row,
                "epoch_metrics_csv": str(od2 / "epoch_metrics.csv"),
            },
        }

    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_json}", flush=True)

    flat_fields = [
        "phase",
        "lr_g",
        "lr_d",
        "lambda_l1",
        "epochs",
        "best_val_l1",
        "final_train_g",
        "final_train_d",
        "final_train_l1",
        "final_val_l1",
        "output_dir",
    ]
    with open(report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        w.writeheader()
        for r in phase1_rows:
            w.writerow({**r, "phase": "phase1", "epochs": str(args.phase1_epochs)})
        if phase2_row:
            w.writerow({**phase2_row, "phase": "phase2_best", "epochs": str(args.phase2_epochs)})
    print(f"Wrote {report_csv}", flush=True)


if __name__ == "__main__":
    main()
