# PyTorch Model Inference Profiler + Optimization Toolkit

A small benchmarking tool that measures and compares **inference** performance of
this project's image-colorization models — before vs. after common optimizations —
and reports the computational-resource metrics needed for the publication's
"what's achievable with limited resources" analysis.

## Why this exists

The publication goal is to evaluate colorization models under **limited compute**.
That requires hard numbers on inference cost, not just output quality. This toolkit
produces those numbers in a consistent, repeatable way and contrasts a lightweight
GAN/CNN-style generator against a heavy diffusion (ControlNet) pipeline so the
resource trade-offs are explicit.

Key idea: **no training or trained weights are required.** Inference latency,
throughput and memory depend on the *architecture, input shape, and precision* —
not on learned weight values. So the profiler runs on randomly-initialized models
(and the pretrained Stable Diffusion cache) and still yields valid timings.

## What it measures

For every `(model, optimization variant, batch size)` combination:

| Metric | Meaning |
|---|---|
| `params_millions` | Parameter count (model size). |
| `cold_ms` | Latency of the very first call (cold start, pre-warmup). |
| `latency_ms_median` / `mean` / `std` / `min` / `max` | Warm latency over `--iters` repeats. |
| `throughput_imgs_per_s` | `batch_size / median_latency`. |
| `peak_mem_mb` | Peak CUDA memory (GPU) or process RSS (CPU). |

CUDA timings are `torch.cuda.synchronize()`-bracketed for accuracy.

## Models profiled

| `--model` | What it is | Inference op |
|---|---|---|
| `cnn` | `ColorizationCNN` (L → ab), the original baseline kept only as a profiling subject | one forward pass |
| `controlnet` | `StableDiffusionControlNetPipeline` (UNet + VAE + text encoder + ControlNet) | full multi-step denoise loop |
| `resnet18` | torchvision reference classifier (`weights=None`, no download) | one forward pass |

The CNN-vs-ControlNet contrast (a ~1 M-param conv net vs. a multi-hundred-M-param
diffusion pipeline) is the point: it shows the optimization toolkit working across a
huge dynamic range and quantifies how much slower diffusion is per image.

## Optimization variants

| Model type | Variants |
|---|---|
| `nn.Module` (cnn, resnet18) | `baseline`, `no_grad`, `inference_mode`, `amp` (CUDA only), `torchscript` |
| pipeline (controlnet) | `fp32`, `fp16`, `bf16` (TorchScript N/A; pipeline already runs under no-grad internally) |

Always-on dimensions: **batching** (`--batch-size` sweep) and **warmup** (cold vs.
warm latency reported separately so the warmup benefit is visible).

## CLI

```bash
# Lightweight models (CPU or GPU)
python profiler/benchmark.py --model cnn      --batch-size 1 8 16 32 --resolution 256
python profiler/benchmark.py --model resnet18 --batch-size 1 8 16 32

# ControlNet (GPU; uses local Stable Diffusion cache)
python profiler/benchmark.py --model controlnet --cond-type softedge \
    --batch-size 1 2 4 --steps 20 --resolution 256 --local-files-only \
    --output-dir runs/profiler/controlnet_softedge
python profiler/benchmark.py --model controlnet --cond-type canny \
    --batch-size 1 2 4 --steps 20 --resolution 256 --local-files-only \
    --output-dir runs/profiler/controlnet_canny
```

Useful flags: `--iters` / `--warmup` (timing budget), `--variants` (subset),
`--device cuda|cpu`, `--steps` (diffusion denoise steps — the dominant latency knob),
`--local-files-only` + `--hf-cache-dir` (force the local SD cache).

## Outputs

Written to `runs/profiler/<model>/` (override with `--output-dir`):

- `results.csv` — one row per `(variant, batch_size)` with all metrics above.
- `latency_vs_batch.png` — median latency vs. batch size, one line per variant.
- `throughput_vs_batch.png` — throughput vs. batch size, one line per variant.
- `baseline_vs_optimized.png` — bar chart at the largest batch size, annotated with
  speedup vs. the baseline/fp32 variant.

## Consistency notes (for the publication)

- Pass the same `--resolution` to every model so latency/memory are comparable
  (CNN defaults to 128, ControlNet to 512; the commands above pin 256).
- ControlNet requires a CUDA GPU in practice; CPU diffusion timing is impractical.
- ControlNet uses the local `models--runwayml--stable-diffusion-v1-5` cache via
  `--local-files-only` (the `runwayml` Hugging Face org is no longer available).

## Files

- `profiler/benchmark.py` — CLI harness: timing, memory, CSV, plots.
- `profiler/models_registry.py` — per-model adapters behind a uniform interface.
