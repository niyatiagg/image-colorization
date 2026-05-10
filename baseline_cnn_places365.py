import argparse
import copy
import csv
import os
import random
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import Places365
from torchvision.transforms import Compose, Resize
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PlacesColorizationDataset(Dataset):
    """
    Returns:
      - l_input: [1, H, W], normalized to [0, 1]
      - ab_target: [2, H, W], normalized to roughly [-1, 1]
    """

    def __init__(self, base_dataset: Dataset, image_size: int = 128):
        self.base_dataset = base_dataset
        self.resize = Compose([Resize((image_size, image_size))])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.base_dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = self.resize(image).convert("RGB")
        image_np = np.array(image)  # RGB uint8 [H, W, 3]

        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0:1] / 255.0  # [0, 1]
        ab_channels = (lab[:, :, 1:3] - 128.0) / 128.0  # approx [-1, 1]

        l_tensor = torch.from_numpy(l_channel).permute(2, 0, 1)  # [1, H, W]
        ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1)  # [2, H, W]
        return l_tensor, ab_tensor


class ColorizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def lab_to_rgb_tensor(l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> torch.Tensor:
    """
    l_tensor: [N,1,H,W] in [0,1]
    ab_tensor: [N,2,H,W] in [-1,1]
    Returns RGB tensor [N,3,H,W] in [0,1]
    """
    l_np = (l_tensor.cpu().numpy() * 255.0).clip(0, 255)
    ab_np = (ab_tensor.cpu().numpy() * 128.0 + 128.0).clip(0, 255)

    rgb_list = []
    for i in range(l_np.shape[0]):
        lab = np.concatenate(
            [l_np[i].transpose(1, 2, 0), ab_np[i].transpose(1, 2, 0)], axis=2
        ).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_list.append(rgb_t)

    return torch.stack(rgb_list, dim=0)


def save_preview(
    model: nn.Module,
    batch_l: torch.Tensor,
    batch_ab: torch.Tensor,
    device: torch.device,
    out_path: str,
    max_images: int = 4,
) -> None:
    model.eval()
    batch_l = batch_l[:max_images].to(device)
    batch_ab = batch_ab[:max_images].to(device)

    pred_ab = model(batch_l)

    gt_rgb = lab_to_rgb_tensor(batch_l, batch_ab)
    pred_rgb = lab_to_rgb_tensor(batch_l, pred_ab)
    gray_rgb = batch_l.repeat(1, 3, 1, 1).cpu()

    rows = []
    for i in range(batch_l.size(0)):
        rows.extend([gray_rgb[i], pred_rgb[i], gt_rgb[i]])
    grid = make_grid(rows, nrow=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def build_dataloaders(
    data_root: str,
    subset_size: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
):
    base = Places365(
        root=data_root,
        split="train-standard",
        small=True,
        download=True,
    )
    total_size = len(base)
    subset_size = min(subset_size, total_size)
    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)
    chosen = indices[:subset_size]

    subset = Subset(base, chosen)
    dataset = PlacesColorizationDataset(subset, image_size=image_size)

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def _param_slug(value: Union[float, int]) -> str:
    """Filesystem-friendly token for hyperparameter values."""
    if isinstance(value, float):
        s = f"{value:.6g}".replace(".", "p").replace("-", "m")
        return s
    return str(value)


def _append_experiment_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _append_epoch_metrics_row(
    out_dir: str, epoch: int, train_l1: float, val_l1: float
) -> None:
    path = os.path.join(out_dir, "epoch_metrics.csv")
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_l1", "val_l1"])
        if not exists:
            w.writeheader()
        w.writerow({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Run: lr={args.lr} batch_size={args.batch_size} "
        f"epochs={args.epochs} subset_size={args.subset_size} -> {args.output_dir}"
    )

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        subset_size=args.subset_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model = ColorizationCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    metrics_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    if os.path.isfile(metrics_path):
        os.remove(metrics_path)
    last_train_l1 = 0.0
    last_val_l1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for l_input, ab_target in train_loader:
            l_input = l_input.to(device)
            ab_target = ab_target.to(device)

            pred_ab = model(l_input)
            loss = criterion(pred_ab, ab_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        model.eval()
        val_running = 0.0
        preview_batch = None
        with torch.no_grad():
            for l_input, ab_target in val_loader:
                l_input = l_input.to(device)
                ab_target = ab_target.to(device)
                pred_ab = model(l_input)
                loss = criterion(pred_ab, ab_target)
                val_running += loss.item()
                if preview_batch is None:
                    preview_batch = (l_input.cpu(), ab_target.cpu())

        val_loss = val_running / max(1, len(val_loader))
        last_train_l1, last_val_l1 = train_loss, val_loss
        _append_epoch_metrics_row(args.output_dir, epoch, train_loss, val_loss)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_l1={train_loss:.4f} val_l1={val_loss:.4f}"
        )

        if preview_batch is not None:
            save_preview(
                model,
                preview_batch[0],
                preview_batch[1],
                device,
                os.path.join(args.output_dir, f"preview_epoch_{epoch:03d}.png"),
            )

        ckpt_path = os.path.join(args.output_dir, "last_model.pt")
        torch.save(model.state_dict(), ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    print(f"Training complete. Outputs written to: {args.output_dir}")
    return {
        "best_val_l1": best_val,
        "output_dir": args.output_dir,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "final_train_l1": last_train_l1,
        "final_val_l1": last_val_l1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline CNN colorization on a small Places365 subset."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/cnn_baseline")
    parser.add_argument("--subset-size", type=int, default=4000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[32],
        help="One or more batch sizes. Multiple values run a grid with --lr.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1e-3],
        help="One or more Adam learning rates. Multiple values run a grid with --batch-size.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    base_args = parse_args()
    base_out = base_args.output_dir
    os.makedirs(base_out, exist_ok=True)
    combos = list(product(base_args.lr, base_args.batch_size))
    log_fields = [
        "lr",
        "batch_size",
        "best_val_l1",
        "final_train_l1",
        "final_val_l1",
        "output_dir",
        "epochs",
        "subset_size",
        "seed",
    ]
    log_path = os.path.join(base_out, "experiment_results.csv")

    for lr, batch_size in combos:
        run_args = copy.deepcopy(base_args)
        run_args.lr = lr
        run_args.batch_size = batch_size
        if len(combos) > 1:
            run_args.output_dir = os.path.join(
                base_out, f"lr_{_param_slug(lr)}_bs_{batch_size}"
            )
        result = train(run_args)
        _append_experiment_csv(
            log_path,
            {
                **result,
                "epochs": base_args.epochs,
                "subset_size": base_args.subset_size,
                "seed": base_args.seed,
            },
            log_fields,
        )
