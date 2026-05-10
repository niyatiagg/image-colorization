# Conditional GAN colorization (`cgan_places365.py`)

## What the code does

The script trains a **Pix2Pix-style conditional GAN** for the same LAB formulation as the baseline CNN:

1. **Generator** `G(L, noise)`: predicts **`ab`** from **`L`** plus one channel of **Gaussian noise** (same spatial size) for stochasticity.
2. **Discriminator** (PatchGAN): sees **`concat(L, ab)`** (3 channels)—real pairs use ground-truth `ab`, fake pairs use `G`’s output—so the critic is **conditioned on luminance**.
3. **Losses**:
   - **Adversarial**: `BCEWithLogitsLoss` so fake `(L, G(L,z))` is classified like real.
   - **Reconstruction**: **`lambda_l1` × L1** between `G(L,z)` and true `ab` (stabilizes training and preserves color layout).
4. **Optimization**: Separate **Adam** optimizers for **G** and **D** with configurable learning rates and betas.
5. **Outputs**: Per-epoch generator/discriminator/L1-style metrics, previews, `last_checkpoint.pt` / `best_checkpoint.pt` (both networks + metadata).

Validation L1 uses **zero noise** so previews and the reported `val_L1` stay comparable across steps.

## Hyperparameter experiments (built in)

You can pass **multiple values** for **`--lr-g`**, **`--lr-d`**, and **`--lambda-l1`**. The script runs the **Cartesian product** of those three lists. When there is more than one triple, each run writes to a **subdirectory** under `--output-dir`.

- **Example (grid)**:

  ```bash
  python cgan_places365.py \
    --output-dir ./runs/cgan_sweep \
    --lr-g 1e-4 2e-4 \
    --lr-d 1e-4 2e-4 \
    --lambda-l1 50 100 \
    --epochs 10
  ```

  This runs **8** trainings: `2 × 2 × 2` combinations. **Be careful**: full grids grow quickly (`n₁ × n₂ × n₃` runs).

- **Typical “one-axis” sweep** (fix two axes with a single value):

  ```bash
  python cgan_places365.py \
    --lr-g 2e-4 \
    --lr-d 2e-4 \
    --lambda-l1 10 50 100 200
  ```

- **Results log**: Rows are appended to  
  `{--output-dir}/experiment_results.csv`  
  with `lr_g`, `lr_d`, `lambda_l1`, `batch_size`, `best_val_l1`, `output_dir`, `epochs`, `subset_size`, and `seed`.

Subfolders for multi-run grids look like:

`lrg_<lr_g>_lrd_<lr_d>_l1_<lambda>/`

### What we are sweeping and why it might (or might not) help

| Hyperparameter | Role | How it might help | How it might *not* help |
|----------------|------|-------------------|-------------------------|
| **`--lr-g`** | Generator Adam LR | If `G` learns too slowly or **mode-collapses** (dull colors), a different LR can improve **color diversity** and adversarial signal. | Too high: **instability**, exploding or oscillating losses; **D** may dominate and `G` stops improving. |
| **`--lr-d`** | Discriminator Adam LR | If `D` is too weak, raising **`lr-d`** can give a sharper training signal; if too strong, lowering it can stop **gradient starvation** of `G`. | **`D` too strong** often yields **washed-out** outputs (high L1, low realism); tuning **`lr-d` alone** rarely fixes bad architecture or too little data. |
| **`--lambda-l1`** | Weight of L1(`ab`) vs GAN loss | **Higher** values push predictions **closer to ground-truth chroma** (safer, blurrier). **Lower** values lean on the GAN for **sharper / more plausible** textures but risk **instability** or color drift. | On a **small subset**, a very low `lambda_l1` may look **noisy**; a very high one can look like **muted regression** (similar to pure L1 CNN). |

**`--batch-size`** is a **single** value per invocation (not part of the GAN grid). Change it on the CLI to apply the same batch size to **every** combination in that run.

## Practical notes

- **Runtime**: GAN training is usually **heavier** than the baseline CNN (two networks, two backward passes per batch).
- **Grid explosion**: Prefer **small sweeps** (e.g. vary only `lambda_l1`, or only `lr-g` with `lr-d` tied) unless you have time for many sequential runs.
- **Monitoring**: Watch **both** `G`/`D` losses and **`val_L1`**; use previews—**lower `val_L1` does not always mean better-looking color** when the GAN term is strong.
