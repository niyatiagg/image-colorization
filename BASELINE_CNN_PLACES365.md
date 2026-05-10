# Baseline CNN colorization (`baseline_cnn_places365.py`)

## What the code does

The script trains a **small convolutional encoder–decoder** to **predict the `a` and `b` channels of LAB color** from the **grayscale `L` channel**, using a random subset of **Places365** (`train-standard`, `small=True` images).

1. **Data**: Each RGB image is resized, converted to LAB with OpenCV, then normalized so `L ∈ [0, 1]` and `ab ∈ [-1, 1]` (approximately).
2. **Model**: `ColorizationCNN` maps a single-channel `L` tensor to a two-channel `ab` prediction with a `Tanh` output.
3. **Loss**: Mean **L1** between predicted and ground-truth `ab`.
4. **Optimization**: **Adam** with a configurable learning rate.
5. **Outputs**: Training/validation L1 per epoch, preview PNGs (grayscale | prediction | ground truth), `best_model.pt` / `last_model.pt`.

## Hyperparameter experiments (built in)

You can pass **multiple values** for **`--lr`** and **`--batch-size`**. The script runs the **Cartesian product** of those lists: each combination gets its own run and, when there is more than one combination, its own subdirectory under `--output-dir`.

- **Example (grid over learning rate and batch size)**:

  ```bash
  python baseline_cnn_places365.py \
    --output-dir ./runs/cnn_sweep \
    --lr 1e-4 3e-4 1e-3 \
    --batch-size 16 32 \
    --epochs 10
  ```

  This runs six trainings: `(lr, batch_size)` ∈ `{1e-4, 3e-4, 1e-3} × {16, 32}`.

- **Single run** (default behavior, one combo):

  ```bash
  python baseline_cnn_places365.py --lr 1e-3 --batch-size 32
  ```

- **Results log**: After each run, a row is appended to  
  `{--output-dir}/experiment_results.csv`  
  with `lr`, `batch_size`, `best_val_l1`, `output_dir`, `epochs`, `subset_size`, and `seed`.

When multiple combinations are used, each run’s artifacts live under:

`{--output-dir}/lr_<lr>_bs_<batch_size>/`

(slugs replace `.` and `-` in folder names for safety).

### What we are sweeping and why it might (or might not) help

| Hyperparameter | Role | How it might help | How it might *not* help |
|----------------|------|-------------------|-------------------------|
| **`--lr`** | Adam step size | Too-small LR can plateau early; a larger LR can fit faster but may diverge or oscillate. Tuning often improves the **first few epochs** and final validation L1. | If the model is **capacity- or data-limited**, changing LR alone may give **small visual gains** after the first 1–2× change. |
| **`--batch-size`** | Samples per gradient step | Larger batches can **stabilize** gradients and better use the GPU; smaller batches add noise that sometimes **helps generalization** (task-dependent). | Very small batches slow training; very large batches need **more VRAM** and can require **retuning LR**. |

Other flags (`--subset-size`, `--epochs`, `--image-size`, etc.) are **fixed per invocation** for all grid points; they are not part of the automatic grid. Change them on the command line if you want those to vary between experiment *batches* (separate script runs).

## Practical notes

- **Grid size**: `len(--lr) × len(--batch-size)` runs execute **sequentially** (one GPU job after another on a single machine).
- **Comparison**: Use `experiment_results.csv` to sort by `best_val_l1`; previews in each run folder show qualitative differences.
