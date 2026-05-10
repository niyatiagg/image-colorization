"""
Conditional GAN (Pix2Pix-style) for LAB colorization on Places365.
Discriminator is conditioned on L by concatenating L with ab channels.
"""

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
        image_np = np.array(image)

        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0:1] / 255.0
        ab_channels = (lab[:, :, 1:3] - 128.0) / 128.0

        l_tensor = torch.from_numpy(l_channel).permute(2, 0, 1)
        ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1)
        return l_tensor, ab_tensor


class ConditionalGenerator(nn.Module):
    """Maps L + noise -> ab; noise gives stochasticity (1 extra channel)."""

    def __init__(self, noise_channels: int = 1):
        super().__init__()
        in_ch = 1 + noise_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
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

    def forward(self, l: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([l, noise], dim=1))


class PatchDiscriminator(nn.Module):
    """PatchGAN: classifies local patches as real/fake given L + ab (3 ch)."""

    def __init__(self) -> None:
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 with stride-2 convs
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            ]
        )

    def forward(self, lab_rgb_like: torch.Tensor) -> torch.Tensor:
        x = lab_rgb_like
        for layer in self.layers:
            x = layer(x)
        return x


@torch.no_grad()
def lab_to_rgb_tensor(l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> torch.Tensor:
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


def concat_l_ab(l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    return torch.cat([l, ab], dim=1)


@torch.no_grad()
def save_preview(
    generator: nn.Module,
    batch_l: torch.Tensor,
    batch_ab: torch.Tensor,
    device: torch.device,
    out_path: str,
    max_images: int = 4,
    noise_std: float = 0.0,
) -> None:
    generator.eval()
    batch_l = batch_l[:max_images].to(device)
    batch_ab = batch_ab[:max_images].to(device)
    noise = torch.randn_like(batch_l) * noise_std if noise_std > 0 else torch.zeros_like(batch_l)

    pred_ab = generator(batch_l, noise)

    gt_rgb = lab_to_rgb_tensor(batch_l, batch_ab)
    pred_rgb = lab_to_rgb_tensor(batch_l, pred_ab)
    gray_rgb = batch_l.repeat(1, 3, 1, 1).cpu()

    rows = []
    for i in range(batch_l.size(0)):
        rows.extend([gray_rgb[i], pred_rgb[i].cpu(), gt_rgb[i]])
    grid = make_grid(rows, nrow=3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(grid, out_path)


def _param_slug(value: Union[float, int]) -> str:
    if isinstance(value, float):
        return f"{value:.6g}".replace(".", "p").replace("-", "m")
    return str(value)


def _append_experiment_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _append_epoch_metrics_row(
    out_dir: str,
    epoch: int,
    train_g: float,
    train_d: float,
    train_l1: float,
    val_l1: float,
) -> None:
    path = os.path.join(out_dir, "epoch_metrics.csv")
    fieldnames = ["epoch", "train_g", "train_d", "train_l1", "val_l1"]
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "epoch": epoch,
                "train_g": train_g,
                "train_d": train_d,
                "train_l1": train_l1,
                "val_l1": val_l1,
            }
        )


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


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Run: lr_g={args.lr_g} lr_d={args.lr_d} lambda_l1={args.lambda_l1} "
        f"batch_size={args.batch_size} epochs={args.epochs} -> {args.output_dir}"
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

    generator = ConditionalGenerator(noise_channels=1).to(device)
    discriminator = PatchDiscriminator().to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    opt_g = optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    opt_d = optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_l1 = float("inf")
    h, w = args.image_size, args.image_size
    metrics_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    if os.path.isfile(metrics_path):
        os.remove(metrics_path)

    last_train_g = last_train_d = last_train_l1 = last_val_l1 = 0.0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        run_g = run_d = run_l1 = 0.0
        n_batches = 0

        for l_input, ab_target in train_loader:
            l_input = l_input.to(device)
            ab_target = ab_target.to(device)
            bsz = l_input.size(0)
            noise = torch.randn(bsz, 1, h, w, device=device)

            real_in = concat_l_ab(l_input, ab_target)
            fake_ab = generator(l_input, noise)
            fake_in = concat_l_ab(l_input, fake_ab.detach())

            # --- Discriminator ---
            opt_d.zero_grad()
            pred_real = discriminator(real_in)
            pred_fake = discriminator(fake_in)
            loss_d_real = criterion_gan(
                pred_real, torch.ones_like(pred_real)
            )
            loss_d_fake = criterion_gan(
                pred_fake, torch.zeros_like(pred_fake)
            )
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # --- Generator ---
            opt_g.zero_grad()
            noise_g = torch.randn(bsz, 1, h, w, device=device)
            fake_ab_g = generator(l_input, noise_g)
            pred_fake_g = discriminator(concat_l_ab(l_input, fake_ab_g))
            loss_g_adv = criterion_gan(
                pred_fake_g, torch.ones_like(pred_fake_g)
            )
            loss_g_l1 = criterion_l1(fake_ab_g, ab_target)
            loss_g = loss_g_adv + float(args.lambda_l1) * loss_g_l1
            loss_g.backward()
            opt_g.step()

            run_g += loss_g.item()
            run_d += loss_d.item()
            run_l1 += loss_g_l1.item()
            n_batches += 1

        train_g = run_g / max(1, n_batches)
        train_d = run_d / max(1, n_batches)
        train_l1 = run_l1 / max(1, n_batches)

        generator.eval()
        val_l1_sum = 0.0
        val_batches = 0
        preview_batch = None
        with torch.no_grad():
            for l_input, ab_target in val_loader:
                l_input = l_input.to(device)
                ab_target = ab_target.to(device)
                noise = torch.zeros_like(l_input)
                pred_ab = generator(l_input, noise)
                val_l1_sum += criterion_l1(pred_ab, ab_target).item()
                val_batches += 1
                if preview_batch is None:
                    preview_batch = (l_input.cpu(), ab_target.cpu())

        val_l1 = val_l1_sum / max(1, val_batches)
        last_train_g, last_train_d, last_train_l1, last_val_l1 = (
            train_g,
            train_d,
            train_l1,
            val_l1,
        )
        _append_epoch_metrics_row(
            args.output_dir, epoch, train_g, train_d, train_l1, val_l1
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"G={train_g:.4f} D={train_d:.4f} train_L1={train_l1:.4f} val_L1={val_l1:.4f}"
        )

        if preview_batch is not None:
            save_preview(
                generator,
                preview_batch[0],
                preview_batch[1],
                device,
                os.path.join(args.output_dir, f"preview_epoch_{epoch:03d}.png"),
                noise_std=0.0,
            )

        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "epoch": epoch,
            },
            os.path.join(args.output_dir, "last_checkpoint.pt"),
        )

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "epoch": epoch,
                    "val_l1": val_l1,
                },
                os.path.join(args.output_dir, "best_checkpoint.pt"),
            )

    print(f"Training complete. Outputs written to: {args.output_dir}")
    return {
        "best_val_l1": best_val_l1,
        "output_dir": args.output_dir,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "lambda_l1": args.lambda_l1,
        "batch_size": args.batch_size,
        "final_train_g": last_train_g,
        "final_train_d": last_train_d,
        "final_train_l1": last_train_l1,
        "final_val_l1": last_val_l1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional GAN colorization (L-conditioned) on Places365 subset."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/cgan_places365")
    parser.add_argument("--subset-size", type=int, default=4000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--lr-g",
        type=float,
        nargs="+",
        default=[2e-4],
        help="One or more generator learning rates. Multiple values grid with --lr-d and --lambda-l1.",
    )
    parser.add_argument(
        "--lr-d",
        type=float,
        nargs="+",
        default=[2e-4],
        help="One or more discriminator learning rates.",
    )
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--lambda-l1",
        type=float,
        nargs="+",
        default=[100.0],
        help="One or more L1 weights on ab (Pix2Pix-style). Multiple values form a Cartesian grid.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    base_args = parse_args()
    base_out = base_args.output_dir
    os.makedirs(base_out, exist_ok=True)
    combos = list(
        product(base_args.lr_g, base_args.lr_d, base_args.lambda_l1)
    )
    log_fields = [
        "lr_g",
        "lr_d",
        "lambda_l1",
        "batch_size",
        "best_val_l1",
        "final_train_g",
        "final_train_d",
        "final_train_l1",
        "final_val_l1",
        "output_dir",
        "epochs",
        "subset_size",
        "seed",
    ]
    log_path = os.path.join(base_out, "experiment_results.csv")

    for lr_g, lr_d, lambda_l1 in combos:
        run_args = copy.deepcopy(base_args)
        run_args.lr_g = lr_g
        run_args.lr_d = lr_d
        run_args.lambda_l1 = lambda_l1
        if len(combos) > 1:
            run_args.output_dir = os.path.join(
                base_out,
                f"lrg_{_param_slug(lr_g)}_lrd_{_param_slug(lr_d)}_l1_{_param_slug(lambda_l1)}",
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
