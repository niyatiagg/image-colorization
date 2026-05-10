"""
ControlNet fine-tuning for image colorization on Places365.

Key design choices (vs. the original "from-scratch" version):
  * Initialise from a pretrained edge ControlNet (default: SoftEdge / HED).
    Edge maps give the diffusion UNet a strong structural prior so the model
    learns *colour fill*, not full re-generation. Other supported preprocessors:
    canny, lineart, scribble, gray (legacy).
  * Train at 512x512 (SD-1.5's native resolution); use AMP, gradient
    checkpointing and gradient accumulation so a single 24 GB GPU is enough.
  * Per-epoch SSIM/PSNR on actually-denoised samples (not just noise MSE).
  * Resume from last_state.pt so a host-side reboot doesn't cost a full run.
  * Optional luminance-preservation post-processing: replace the predicted
    image's L (CIE Lab) with the input grayscale L, so structure is locked.
"""

import argparse
import csv
import os
import random
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import Places365
from torchvision.transforms import Compose, Resize
from torchvision.utils import make_grid, save_image

try:
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        StableDiffusionControlNetPipeline,
        UNet2DConditionModel,
        UniPCMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependencies. Install: pip install diffusers transformers accelerate"
    ) from exc


COND_TO_CONTROLNET = {
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "gray": None,  # no pretrained head; init from UNet copy
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Conditioning preprocessors                                                  #
# --------------------------------------------------------------------------- #


class _GrayCond:
    """Replicates the grayscale luminance into 3 channels."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"))
        gray = arr.mean(axis=2).astype(np.uint8)
        return Image.fromarray(np.stack([gray] * 3, axis=2))


class _CannyCond:
    def __init__(self, low: int = 100, high: int = 200) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)
        return Image.fromarray(np.stack([edges] * 3, axis=2))


class _AuxCond:
    """Wraps any controlnet_aux detector that takes/returns a PIL image."""

    def __init__(self, detector_cls_name: str) -> None:
        try:
            from controlnet_aux import HEDdetector, LineartDetector, PidiNetDetector
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "controlnet_aux not installed. Install with: pip install controlnet_aux"
            ) from exc

        registry = {
            "HEDdetector": HEDdetector,
            "LineartDetector": LineartDetector,
            "PidiNetDetector": PidiNetDetector,
        }
        cls = registry[detector_cls_name]
        self.detector = cls.from_pretrained("lllyasviel/Annotators")
        self._is_pidi = detector_cls_name == "PidiNetDetector"

    def __call__(self, img: Image.Image) -> Image.Image:
        out = (
            self.detector(img, safe=True, scribble=True)
            if self._is_pidi
            else self.detector(img)
        )
        if isinstance(out, np.ndarray):
            out = Image.fromarray(out)
        return out.convert("RGB")


def build_cond_processor(cond_type: str) -> Callable[[Image.Image], Image.Image]:
    if cond_type == "gray":
        return _GrayCond()
    if cond_type == "canny":
        return _CannyCond()
    if cond_type == "softedge":
        return _AuxCond("HEDdetector")
    if cond_type == "lineart":
        return _AuxCond("LineartDetector")
    if cond_type == "scribble":
        return _AuxCond("PidiNetDetector")
    raise ValueError(f"Unknown cond-type: {cond_type}")


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class PlacesColorCondDataset(Dataset):
    """
    Returns:
      - cond_rgb:   [3, H, W] in [0, 1]  (controlnet conditioning)
      - target_rgb: [3, H, W] in [-1, 1] (VAE input)
      - gray_rgb:   [3, H, W] in [0, 1]  (kept for preview/luminance lock)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        image_size: int,
        cond_processor: Callable[[Image.Image], Image.Image],
    ):
        self.base_dataset = base_dataset
        self.resize = Compose([Resize((image_size, image_size))])
        self.cond_processor = cond_processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, _ = self.base_dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.resize(image).convert("RGB")

        cond_img = self.cond_processor(image)
        if cond_img.size != image.size:
            cond_img = cond_img.resize(image.size, Image.BILINEAR)

        rgb = np.asarray(image).astype(np.float32) / 255.0
        cond = np.asarray(cond_img.convert("RGB")).astype(np.float32) / 255.0
        gray = rgb.mean(axis=2, keepdims=True).repeat(3, axis=2)

        target_t = torch.from_numpy(rgb).permute(2, 0, 1) * 2.0 - 1.0
        cond_t = torch.from_numpy(cond).permute(2, 0, 1)  # [0, 1]
        gray_t = torch.from_numpy(gray).permute(2, 0, 1)  # [0, 1]
        return cond_t, target_t, gray_t


def build_dataloaders(
    args: argparse.Namespace,
    cond_processor: Callable[[Image.Image], Image.Image],
) -> Tuple[DataLoader, DataLoader]:
    extracted = os.path.join(args.data_root, "data_256_standard")
    needs_download = not os.path.isdir(extracted)
    base = Places365(
        root=args.data_root,
        split="train-standard",
        small=True,
        download=needs_download,
    )
    total_size = len(base)
    subset_size = min(args.subset_size, total_size)
    indices = list(range(total_size))
    random.Random(args.seed).shuffle(indices)
    chosen = indices[:subset_size]

    subset = Subset(base, chosen)
    dataset = PlacesColorCondDataset(subset, args.image_size, cond_processor)

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def build_text_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        ["a realistic color photo"] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


def luminance_lock(pred_rgb01: np.ndarray, gray_rgb01: np.ndarray) -> np.ndarray:
    """Replace L (CIE Lab) of pred with L of gray; keeps structure intact."""
    pred_u8 = (pred_rgb01.clip(0, 1) * 255.0).astype(np.uint8)
    gray_u8 = (gray_rgb01.clip(0, 1) * 255.0).astype(np.uint8)
    pred_lab = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2LAB)
    gray_lab = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2LAB)
    pred_lab[:, :, 0] = gray_lab[:, :, 0]
    out = cv2.cvtColor(pred_lab, cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Both inputs HxWx3 in [0, 1]."""
    from skimage.metrics import (
        peak_signal_noise_ratio,
        structural_similarity,
    )

    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(target, pred, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


@torch.no_grad()
def sample_and_score(
    pipe: StableDiffusionControlNetPipeline,
    cond_batch: torch.Tensor,
    target_batch: torch.Tensor,
    gray_batch: torch.Tensor,
    args: argparse.Namespace,
    out_path: Optional[str] = None,
) -> Tuple[float, float]:
    cond_batch = cond_batch[: args.preview_images]
    target_batch = target_batch[: args.preview_images]
    gray_batch = gray_batch[: args.preview_images]

    cond_imgs = [
        Image.fromarray((c.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        for c in cond_batch
    ]
    generated = pipe(
        prompt=["a realistic color photo"] * len(cond_imgs),
        image=cond_imgs,
        num_inference_steps=args.preview_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    ).images

    psnrs, ssims = [], []
    rows = []
    for i, gen_img in enumerate(generated):
        pred = np.asarray(gen_img).astype(np.float32) / 255.0
        gray_np = gray_batch[i].permute(1, 2, 0).cpu().numpy()
        if args.preserve_luminance:
            pred = luminance_lock(pred, gray_np)
        target = ((target_batch[i].clamp(-1, 1).cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        p, s = compute_psnr_ssim(pred, target)
        psnrs.append(p)
        ssims.append(s)

        if out_path is not None:
            cond_t = cond_batch[i].clamp(0, 1).cpu()
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).float()
            target_t = torch.from_numpy(target).permute(2, 0, 1).float()
            rows.extend([cond_t, pred_t, target_t])

    if out_path is not None and rows:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(make_grid(rows, nrow=3), out_path)

    return float(np.mean(psnrs)), float(np.mean(ssims))


def append_metrics(path: str, row: Dict[str, float]) -> None:
    fieldnames = ["epoch", "train_mse", "val_mse", "sample_psnr", "sample_ssim"]
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_optimizer(
    args: argparse.Namespace, params
) -> torch.optim.Optimizer:
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install bitsandbytes for 8-bit Adam: pip install bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv = os.path.join(args.output_dir, "epoch_metrics.csv")

    print(f"Device: {device}")
    print(
        f"Config: cond_type={args.cond_type}, image_size={args.image_size}, "
        f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
        f"epochs={args.epochs}, amp={use_amp}, ckpt={args.gradient_checkpointing}"
    )

    hf_kwargs = {
        "cache_dir": args.hf_cache_dir or None,
        "token": args.hf_token or None,
        "local_files_only": args.local_files_only,
    }
    print(f"Loading base model: {args.base_model_id}")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_id, subfolder="scheduler", **hf_kwargs
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.base_model_id, subfolder="tokenizer", **hf_kwargs
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_id, subfolder="text_encoder", **hf_kwargs
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.base_model_id, subfolder="vae", **hf_kwargs
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model_id, subfolder="unet", **hf_kwargs
    ).to(device)

    controlnet_id = args.controlnet_id or COND_TO_CONTROLNET[args.cond_type]
    if controlnet_id is None:
        print("Initialising ControlNet from UNet copy (gray cond, training from scratch).")
        controlnet = ControlNetModel.from_unet(unet).to(device)
    else:
        print(f"Loading pretrained ControlNet: {controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, **hf_kwargs
        ).to(device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        try:
            controlnet.enable_gradient_checkpointing()
        except AttributeError:
            pass
    controlnet.train()

    print(f"Building cond processor: {args.cond_type}")
    cond_processor = build_cond_processor(args.cond_type)
    train_loader, val_loader = build_dataloaders(args, cond_processor)
    print(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    optimizer = build_optimizer(args, controlnet.parameters())
    amp_dtype = torch.bfloat16 if (use_amp and args.bf16) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    vae_scale = getattr(vae.config, "scaling_factor", 0.18215)

    start_epoch = 1
    best_val = float("inf")
    state_path = os.path.join(args.output_dir, "training_state.pt")

    if args.resume and os.path.isfile(state_path):
        print(f"Resuming from {state_path}")
        ckpt = torch.load(state_path, map_location=device)
        controlnet.load_state_dict(ckpt["controlnet"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed at epoch {start_epoch}, best_val={best_val:.5f}")
    elif os.path.isfile(metrics_csv):
        os.remove(metrics_csv)

    preview_pipe: Optional[StableDiffusionControlNetPipeline] = None

    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if use_amp
        else nullcontext
    )

    for epoch in range(start_epoch, args.epochs + 1):
        controlnet.train()
        train_loss = 0.0
        n_train = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            cond_rgb, target_rgb, _gray_rgb = batch
            cond_rgb = cond_rgb.to(device, non_blocking=True)
            target_rgb = target_rgb.to(device, non_blocking=True)
            bsz = target_rgb.size(0)
            encoder_hidden_states = build_text_embeddings(
                tokenizer, text_encoder, bsz, device
            )

            with torch.no_grad():
                latents = vae.encode(target_rgb).latent_dist.sample() * vae_scale
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast_ctx():
                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond_rgb,
                    return_dict=False,
                )
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_loss += loss.item() * args.grad_accum
            n_train += 1

            if (step + 1) % args.grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if args.max_train_steps and n_train >= args.max_train_steps:
                break

        avg_train = train_loss / max(1, n_train)

        controlnet.eval()
        val_loss = 0.0
        n_val = 0
        preview_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        with torch.no_grad():
            for cond_rgb, target_rgb, gray_rgb in val_loader:
                cond_rgb = cond_rgb.to(device, non_blocking=True)
                target_rgb = target_rgb.to(device, non_blocking=True)
                bsz = target_rgb.size(0)
                encoder_hidden_states = build_text_embeddings(
                    tokenizer, text_encoder, bsz, device
                )

                latents = vae.encode(target_rgb).latent_dist.sample() * vae_scale
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with autocast_ctx():
                    down_samples, mid_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=cond_rgb,
                        return_dict=False,
                    )
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_samples,
                        mid_block_additional_residual=mid_sample,
                    ).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                val_loss += loss.item()
                n_val += 1
                if preview_batch is None:
                    preview_batch = (cond_rgb.cpu(), target_rgb.cpu(), gray_rgb.cpu())

        avg_val = val_loss / max(1, n_val)

        sample_psnr = sample_ssim = float("nan")
        if preview_batch is not None:
            if preview_pipe is None:
                preview_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    args.base_model_id,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    controlnet=controlnet,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    **hf_kwargs,
                ).to(device)
                preview_pipe.scheduler = UniPCMultistepScheduler.from_config(
                    preview_pipe.scheduler.config
                )
                preview_pipe.set_progress_bar_config(disable=True)

            do_save = (epoch % args.preview_every == 0) or (epoch == args.epochs)
            preview_path = (
                os.path.join(args.output_dir, f"preview_epoch_{epoch:03d}.png")
                if do_save
                else None
            )
            sample_psnr, sample_ssim = sample_and_score(
                preview_pipe,
                preview_batch[0],
                preview_batch[1],
                preview_batch[2],
                args,
                preview_path,
            )

        append_metrics(
            metrics_csv,
            {
                "epoch": epoch,
                "train_mse": avg_train,
                "val_mse": avg_val,
                "sample_psnr": sample_psnr,
                "sample_ssim": sample_ssim,
            },
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_mse={avg_train:.5f} val_mse={avg_val:.5f} "
            f"psnr={sample_psnr:.2f} ssim={sample_ssim:.3f}"
        )

        is_best = avg_val < best_val
        if is_best:
            best_val = avg_val

        torch.save(
            {
                "controlnet": controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "best_val": best_val,
                "args": vars(args),
            },
            state_path,
        )
        torch.save(
            controlnet.state_dict(),
            os.path.join(args.output_dir, "last_controlnet.pt"),
        )
        if is_best:
            torch.save(
                controlnet.state_dict(),
                os.path.join(args.output_dir, "best_controlnet.pt"),
            )

    if preview_pipe is not None:
        del preview_pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Best val_mse={best_val:.5f}. Outputs: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ControlNet recolour fine-tuning on Places365."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/controlnet_recolor")
    parser.add_argument("--base-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--cond-type",
        type=str,
        default="softedge",
        choices=list(COND_TO_CONTROLNET.keys()),
        help="Conditioning preprocessor (default: softedge / HED).",
    )
    parser.add_argument(
        "--controlnet-id",
        type=str,
        default="",
        help="Override the pretrained ControlNet checkpoint id.",
    )
    parser.add_argument("--subset-size", type=int, default=4000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Optional cap on optimizer steps per epoch (0 = unlimited).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument("--preview-steps", type=int, default=20)
    parser.add_argument(
        "--controlnet-conditioning-scale", type=float, default=1.0,
        help="Strength of ControlNet at inference (1.0 = trust the cond fully).",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.5,
        help="CFG scale at preview inference. Lower keeps the cond stream dominant.",
    )
    parser.add_argument(
        "--preserve-luminance", action="store_true", default=True,
        help="Replace L-channel of pred with L of input grayscale (locks structure).",
    )
    parser.add_argument(
        "--no-preserve-luminance", dest="preserve_luminance", action="store_false",
    )
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Mixed precision (default on, requires CUDA).",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--bf16", action="store_true", default=True,
        help="Use bfloat16 (default; safer than fp16 for SD-VAE on Ampere+).",
    )
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument(
        "--use-8bit-adam", action="store_true",
        help="Use bitsandbytes AdamW8bit (saves ~6 GB; requires bitsandbytes).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from training_state.pt in --output-dir if present.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (or HF_TOKEN env var).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", ""),
        help="Optional HF cache directory.",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Only load from local HF cache.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
