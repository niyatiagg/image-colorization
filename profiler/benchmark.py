#!/usr/bin/env python3
"""PyTorch Model Inference Profiler + Optimization Toolkit.

Benchmarks inference latency, throughput, peak memory and batch-size scaling for
the image-colorization models in this repo (and an optional ResNet18 reference),
comparing a baseline forward pass against optimization variants:

    - torch.no_grad() / torch.inference_mode()
    - mixed precision (CUDA autocast)
    - TorchScript tracing (nn.Module models)
    - precision levels fp32 / fp16 / bf16 (ControlNet pipeline)
    - batching (sweep over --batch-size)
    - model warmup (cold-start latency reported separately)

No training or trained weights are required: inference cost depends on the
architecture, input shape and precision, not on learned weight values.

Examples
--------
    python benchmark.py --model cnn        --batch-size 1 8 16 32
    python benchmark.py --model resnet18   --batch-size 32
    python benchmark.py --model controlnet --batch-size 1 2 --steps 20 \
        --cond-type softedge --local-files-only

Outputs (under --output-dir, default runs/profiler/<model>/):
    results.csv
    latency_vs_batch.png
    throughput_vs_batch.png
    baseline_vs_optimized.png
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import resource
import time
from typing import Any, Callable, Dict, List

import torch

from models_registry import build_adapter


# --------------------------------------------------------------------------- #
# Measurement                                                                 #
# --------------------------------------------------------------------------- #


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    # Linux ru_maxrss is in kilobytes; rough process peak RSS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def time_calls(
    call: Callable[[Any], Any],
    inputs: Any,
    iters: int,
    warmup: int,
    device: torch.device,
) -> Dict[str, float]:
    """Measure one configured inference call.

    Returns cold-start latency (first ever call), warm latency stats over
    ``iters`` repeats, and peak memory during the timed window.
    """
    # Cold start (pre-warmup) to quantify the warmup benefit.
    _sync(device)
    t0 = time.perf_counter()
    call(inputs)
    _sync(device)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        call(inputs)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples: List[float] = []
    for _ in range(iters):
        _sync(device)
        t = time.perf_counter()
        call(inputs)
        _sync(device)
        samples.append((time.perf_counter() - t) * 1000.0)

    samples.sort()
    n = len(samples)
    median = samples[n // 2]
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / n
    return {
        "cold_ms": cold_ms,
        "latency_ms_median": median,
        "latency_ms_mean": mean,
        "latency_ms_std": var ** 0.5,
        "latency_ms_min": samples[0],
        "latency_ms_max": samples[-1],
        "peak_mem_mb": _peak_mem_mb(device),
    }


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def make_plots(rows: List[Dict[str, Any]], out_dir: str, model_name: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({r["variant"] for r in rows})
    batch_sizes = sorted({r["batch_size"] for r in rows})
    paths: List[str] = []

    def series(variant: str, key: str):
        pts = [
            (r["batch_size"], r[key])
            for r in rows
            if r["variant"] == variant and r[key] == r[key]  # drop NaN
        ]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    # 1) Latency vs batch size.
    plt.figure(figsize=(7, 4.5))
    for v in variants:
        xs, ys = series(v, "latency_ms_median")
        if xs:
            plt.plot(xs, ys, marker="o", label=v)
    plt.xlabel("batch size")
    plt.ylabel("median latency (ms)")
    plt.title(f"{model_name}: latency vs batch size")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    p = os.path.join(out_dir, "latency_vs_batch.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # 2) Throughput vs batch size.
    plt.figure(figsize=(7, 4.5))
    for v in variants:
        xs, ys = series(v, "throughput_imgs_per_s")
        if xs:
            plt.plot(xs, ys, marker="o", label=v)
    plt.xlabel("batch size")
    plt.ylabel("throughput (images/s)")
    plt.title(f"{model_name}: throughput vs batch size")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    p = os.path.join(out_dir, "throughput_vs_batch.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # 3) Baseline vs optimized at the largest batch size.
    target_bs = batch_sizes[-1]
    bar_rows = [r for r in rows if r["batch_size"] == target_bs]
    bar_rows.sort(key=lambda r: r["latency_ms_median"])
    if bar_rows:
        plt.figure(figsize=(7, 4.5))
        names = [r["variant"] for r in bar_rows]
        lat = [r["latency_ms_median"] for r in bar_rows]
        bars = plt.bar(names, lat, color="#4C72B0")
        base = next((r for r in bar_rows if r["variant"] in ("baseline", "fp32")), None)
        if base is not None:
            for rect, r in zip(bars, bar_rows):
                speedup = base["latency_ms_median"] / r["latency_ms_median"]
                plt.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        plt.ylabel("median latency (ms)")
        plt.title(f"{model_name}: baseline vs optimized (batch={target_bs})")
        plt.xticks(rotation=20)
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        p = os.path.join(out_dir, "baseline_vs_optimized.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(p)

    return paths


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["cnn", "controlnet", "resnet18"])
    p.add_argument("--batch-size", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument("--resolution", type=int, default=0, help="0 = model default.")
    p.add_argument("--variants", type=str, nargs="*", default=None,
                   help="Subset of variants; default = all supported by the model.")
    p.add_argument("--iters", type=int, default=20, help="Timed repeats per config.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iters before timing.")
    p.add_argument("--device", type=str, default="", help="cuda / cpu (auto by default).")
    p.add_argument("--output-dir", type=str, default="", help="Default runs/profiler/<model>.")

    # ControlNet-specific.
    p.add_argument("--steps", type=int, default=20, help="ControlNet denoise steps.")
    p.add_argument("--cond-type", type=str, default="softedge")
    p.add_argument("--base-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet-id", type=str, default="")
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    p.add_argument("--hf-cache-dir", type=str, default=os.environ.get("HF_HOME", ""))
    p.add_argument("--local-files-only", action="store_true")

    args = p.parse_args()
    if args.resolution == 0:
        args.resolution = None
    return args


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.output_dir or os.path.join("runs", "profiler", args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}  ({'CUDA ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor() or 'cpu'})")
    print(f"Building adapter: {args.model}")
    adapter = build_adapter(args, device)

    variants = args.variants or adapter.default_variants()
    param_count = adapter.param_count()
    print(f"Parameters: {param_count / 1e6:.2f} M")
    print(f"Variants: {variants}")
    print(f"Batch sizes: {args.batch_size}")

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        try:
            call = adapter.prepare(variant)
        except ValueError as exc:
            print(f"[skip] variant '{variant}': {exc}")
            continue

        for bs in args.batch_size:
            tag = f"{args.model}/{variant}/bs={bs}"
            try:
                inputs = adapter.make_inputs(bs)
                stats = time_calls(call, inputs, args.iters, args.warmup, device)
            except RuntimeError as exc:
                print(f"[oom/err] {tag}: {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            throughput = bs / (stats["latency_ms_median"] / 1000.0)
            row = {
                "model": args.model,
                "variant": variant,
                "batch_size": bs,
                "resolution": args.resolution or "default",
                "params_millions": round(param_count / 1e6, 4),
                "throughput_imgs_per_s": round(throughput, 3),
                "device": device.type,
                **{k: round(v, 4) for k, v in stats.items()},
            }
            if args.model == "controlnet":
                row["steps"] = args.steps
            rows.append(row)
            print(
                f"  {tag:>32}  "
                f"median={stats['latency_ms_median']:8.2f}ms  "
                f"cold={stats['cold_ms']:8.2f}ms  "
                f"thrpt={throughput:8.2f} img/s  "
                f"peak_mem={stats['peak_mem_mb']:8.1f}MB"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("No successful measurements; nothing to write.")
        return

    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    plot_paths = make_plots(rows, out_dir, args.model)
    for p in plot_paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
