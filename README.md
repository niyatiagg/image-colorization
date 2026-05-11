## Comparative Image Colorization on Places365

Three approaches to grayscale-to-color image translation, trained and compared on a subset of the **Places365** dataset:

1. **CNN baseline** — small encoder–decoder predicting `ab` from `L` in CIE Lab, supervised with L1.
2. **Conditional GAN** — Pix2Pix-style generator + PatchGAN discriminator (also `L → ab`), with L1 + adversarial loss.
3. **ControlNet recolour** — fine-tunes a pretrained Stable Diffusion 1.5 ControlNet (SoftEdge / HED) so the diffusion UNet **fills colour while a structural edge prior preserves the input layout**.

### Highlights

- **CNN baseline** for fast, deterministic colorization.
- **Conditional GAN** (Pix2Pix-style) for sharper, more plausible colour distributions.
- **ControlNet (SD 1.5 + HED softedge)** for diffusion-based colorization that respects input structure via a pretrained edge ControlNet, with optional **luminance lock** (replace predicted L with input L in CIE Lab) so structure is preserved by construction.
- Per-epoch **PSNR / SSIM** (and L1 / GAN losses where applicable), preview grids, and CSV metrics.
- **Hyperparameter sweeps** built into both CNN and cGAN scripts (Cartesian grid over LR / batch / `lambda_l1`).
- A **comparison runner** (`run_full_comparison_study.py`) that orchestrates all three methods end-to-end.

### Repo layout

```
image-colorization/
  README.md                              # this file
  BASELINE_CNN_PLACES365.md              # CNN script + sweep details
  CGAN_PLACES365.md                      # cGAN script + sweep details
  baseline_cnn_places365.py              # CNN training (with built-in HP sweep)
  cgan_places365.py                      # cGAN training (with built-in HP sweep)
  controlnet_recolor_places365.py        # ControlNet fine-tuning (SoftEdge / Canny / Lineart / Scribble / Gray)
  run_cgan_hyperparameter_study.py       # cGAN-only sweep helper
  run_full_comparison_study.py           # Runs all three methods + aggregates results
  requirements-controlnet.txt            # Pinned stack for the diffusion path
  runs/                                  # Per-method output dirs (CSV metrics + preview PNGs tracked, weights ignored)
```

Not in git: `.venv/`, `data/` (Places365), `models--runwayml--stable-diffusion-v1-5/` (HF cache), `runs/**/*.pt` (model checkpoints + optimizer state).

### Methods at a glance

| Method | Model space | Loss | Output |
|---|---|---|---|
| CNN baseline | `L → ab` (small CNN) | L1 | Predicted `ab` recombined with input `L` |
| cGAN (Pix2Pix) | `L (+ noise) → ab`, PatchGAN `D` on `concat(L, ab)` | BCE adv. + `λ·L1` | Predicted `ab` recombined with input `L` |
| ControlNet (SD 1.5) | Diffusion in latent space, conditioned on a **HED softedge map** of the grayscale input | ε-prediction MSE in latent space | Sampled RGB image, optionally **luminance-locked** to input `L` |

The first two are deterministic regression-style colorizers in **Lab space** (so structure is trivially preserved). The third is a generative diffusion model conditioned on an edge map, which is why structural fidelity comes from (a) the pretrained softedge ControlNet and (b) the optional luminance-lock post-processing.

For the ControlNet, supported `--cond-type` values are `softedge` (default, HED), `canny`, `lineart`, `scribble`, and `gray` (legacy). Each one auto-selects a matching pretrained ControlNet head (`lllyasviel/control_v11p_sd15_*`).

### Reproducing

This pipeline has two installs: a **driver / system** install (only needed once per machine) and a **Python** install (per project venv).

#### 1. NVIDIA driver — only if running ControlNet on a GPU

On a fresh Ubuntu 24.04 AWS instance with a `linux-aws` kernel, the precompiled NVIDIA module path avoids `dkms` headaches:

```bash
sudo apt update
sudo apt install -y \
  linux-modules-nvidia-535-server-$(uname -r) \
  nvidia-headless-no-dkms-535-server \
  nvidia-utils-535-server
sudo modprobe nvidia
sudo nvidia-modprobe -c 0 -u   # creates /dev/nvidia* without a reboot
nvidia-smi                     # must show your GPU
```

The CNN baseline and cGAN run fine on CPU (slowly) or on any CUDA GPU; only the ControlNet path strictly needs a recent NVIDIA GPU (24 GB VRAM recommended for SD 1.5 at 384–512 px with gradient checkpointing).

#### 2. Python environment

```bash
python -m venv .venv && source .venv/bin/activate

# For ControlNet you need the pinned stack (see notes below):
pip install -r requirements-controlnet.txt
```

The pin file in particular forces `torch==2.4.1+cu121`, `diffusers==0.31.0`, and `transformers==4.46.3`. **Do not bump `diffusers` above 0.31 unless you also bump `torch` to ≥ 2.6.** `diffusers ≥ 0.36` registers a Flash-Attention 3 custom op whose type hints (`float | None`) crash `torch 2.4`'s `infer_schema` at import time.

`torch==2.4.1+cu121` is required because Ubuntu's stock NVIDIA 535 driver only supports up to CUDA 12.2; the default PyPI `torch` wheels currently target newer CUDA and silently fall back to CPU.

#### 3. Data

`torchvision.datasets.Places365` will download the small (256×256) split to `./data/data_256_standard/` on first use:

```python
from torchvision.datasets import Places365
Places365(root="./data", split="train-standard", small=True, download=True)
```

The dataset is ~50 GB, so this is gitignored.

### Running each method

#### CNN baseline

```bash
python baseline_cnn_places365.py \
  --output-dir ./runs/cnn_baseline \
  --subset-size 4000 --epochs 20 \
  --lr 1e-3 --batch-size 32
```

See `BASELINE_CNN_PLACES365.md` for the built-in `--lr` × `--batch-size` Cartesian sweep.

#### Conditional GAN

```bash
python cgan_places365.py \
  --output-dir ./runs/cgan_places365 \
  --subset-size 4000 --epochs 20 \
  --lr-g 2e-4 --lr-d 2e-4 --lambda-l1 100 \
  --batch-size 32
```

See `CGAN_PLACES365.md` for the built-in `--lr-g` × `--lr-d` × `--lambda-l1` sweep.

#### ControlNet (SD 1.5 + SoftEdge / HED)

```bash
nohup python -u controlnet_recolor_places365.py \
  --output-dir ./runs/controlnet_softedge \
  --cond-type softedge \
  --subset-size 8000 --epochs 10 \
  --image-size 384 --batch-size 2 --grad-accum 4 \
  --num-workers 2 --preview-every 1 --resume \
  > runs/controlnet_softedge.log 2>&1 &
```

Key flags:

- `--cond-type {softedge,canny,lineart,scribble,gray}` — preprocessor + matching pretrained ControlNet head.
- `--image-size 384 --batch-size 2 --grad-accum 4` — effective batch 8; fits comfortably on a 24 GB A10G with `--amp` (bf16) and `--gradient-checkpointing` (both default on).
- `--preserve-luminance` (default on) — replaces predicted `L` with input `L` in CIE Lab so structure is locked.
- `--resume` — picks up from `runs/<dir>/training_state.pt` if a prior run was interrupted.

Per-epoch metrics (`train_mse`, `val_mse`, `sample_psnr`, `sample_ssim`) land in `runs/<dir>/epoch_metrics.csv`. Preview grids (`cond | prediction | ground truth`) land in `runs/<dir>/preview_epoch_NNN.png`.

#### Full comparison

```bash
python run_full_comparison_study.py --output-root ./runs/final_comparison
```

This runs the CNN, the cGAN study, and the ControlNet recolour, then aggregates best metrics into a single CSV.

### Project report (LaTeX + Word)

- **`report.tex`** — IEEE-style conference paper (abstract through conclusion, bibliography, **Appendix A** with qualitative comparison figures embedded from `runs/…`).
- **`report.docx`** — Word export with the same narrative, tables, and appendix images. Regenerate after changing results or swapping preview PNGs:

  ```bash
  pip install python-docx   # also listed in requirements-controlnet.txt
  python build_report_docx.py
  ```

  Build the PDF (requires a TeX distribution, e.g. `texlive-latex-extra`):

  ```bash
  pdflatex -interaction=nonstopmode report.tex
  pdflatex -interaction=nonstopmode report.tex
  ```

### Tech stack

- **Python 3.12**, **PyTorch 2.4 (CUDA 12.1 wheels)**
- **CNN baseline** (encoder–decoder)
- **Conditional GAN** (Pix2Pix-style + PatchGAN)
- **Stable Diffusion 1.5** + **ControlNet (SoftEdge / HED)** via `diffusers 0.31` and `transformers 4.46`
- `controlnet_aux` (HED, lineart, scribble, canny preprocessors)
- **OpenCV**, **NumPy**, **scikit-image** (PSNR / SSIM)
- **torchvision** (`Places365` dataset)
- Trained on **AWS EC2 g5.2xlarge** (NVIDIA A10G, 24 GB VRAM)

### Notes / lessons learned

- **diffusers + torch version coupling is real.** The current pin (`diffusers==0.31`, `torch==2.4.1+cu121`) is what works for SD 1.5 ControlNet fine-tuning on a 24 GB GPU with a CUDA 12.x driver. Bumping either one in isolation breaks imports.
- **Structural preservation in diffusion colorization comes from three places**: the pretrained edge ControlNet (we use SoftEdge), the conditioning preprocessor (HED edges of the grayscale input), and the optional luminance-lock (CIE Lab `L` swap at the very end). All three matter.
- **For colorization of natural photos, SoftEdge / HED outperforms Canny** — HED suppresses noisy texture edges that would otherwise confuse the model, while still capturing semantic contours.
- Training from scratch with `ControlNetModel.from_unet(...)` does **not** work on a small dataset — the diffusion prior dominates and the model "regenerates" the scene instead of recolouring it. Always start from a pretrained ControlNet head when fine-tuning at small scale.
