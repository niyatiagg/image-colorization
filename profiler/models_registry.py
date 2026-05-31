"""Model adapters for the inference profiler.

Each adapter exposes a uniform interface the benchmark harness can drive:

    adapter.param_count() -> int
    adapter.default_variants() -> list[str]
    adapter.make_inputs(batch_size) -> Any        # one batch of model inputs
    adapter.prepare(variant) -> Callable[[Any], Any]  # one configured inference call

The returned callable runs exactly ONE inference and owns all variant semantics
(``torch.no_grad`` / ``inference_mode`` / autocast / TorchScript / dtype), so the
harness only has to time ``call(inputs)`` and synchronise CUDA around it.

Adapters covered:
  - ``cnn``        : ColorizationCNN from baseline_cnn_places365.py (L -> ab)
  - ``controlnet`` : StableDiffusionControlNetPipeline (multi-step denoise loop)
  - ``resnet18``   : torchvision reference classifier (weights=None, no download)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, List

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _amp_dtype() -> torch.dtype:
    """Preferred low-precision dtype for CUDA autocast."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------------------- #
# Generic nn.Module adapter (covers CNN and ResNet18)                         #
# --------------------------------------------------------------------------- #


class TorchModuleAdapter:
    """Profiles any single-tensor-in nn.Module (CNN colorizer, ResNet18, ...)."""

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        input_shape: tuple,
        device: torch.device,
    ) -> None:
        self.name = name
        self.device = device
        self.input_shape = input_shape  # per-sample shape, e.g. (1, H, W)
        self.model = model.to(device).eval()
        self._traced: torch.nn.Module | None = None

    def param_count(self) -> int:
        return _count_params(self.model)

    def default_variants(self) -> List[str]:
        variants = ["baseline", "no_grad", "inference_mode", "torchscript"]
        if self.device.type == "cuda":
            variants.insert(3, "amp")
        return variants

    def make_inputs(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, *self.input_shape, device=self.device)

    def _build_traced(self) -> torch.nn.Module:
        if self._traced is None:
            example = torch.randn(1, *self.input_shape, device=self.device)
            with torch.no_grad():
                traced = torch.jit.trace(self.model, example)
            try:
                traced = torch.jit.freeze(traced.eval())
            except Exception:  # noqa: BLE001 - freeze is best-effort
                pass
            self._traced = traced
        return self._traced

    def prepare(self, variant: str) -> Callable[[Any], Any]:
        model = self.model

        if variant == "baseline":
            def call(x: torch.Tensor) -> torch.Tensor:
                return model(x)

        elif variant == "no_grad":
            def call(x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return model(x)

        elif variant == "inference_mode":
            def call(x: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return model(x)

        elif variant == "amp":
            if self.device.type != "cuda":
                raise ValueError("amp variant requires CUDA")
            dtype = _amp_dtype()

            def call(x: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    return model(x)

        elif variant == "torchscript":
            traced = self._build_traced()

            def call(x: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return traced(x)

        else:
            raise ValueError(f"Unknown variant for {self.name}: {variant}")

        return call


# --------------------------------------------------------------------------- #
# ControlNet pipeline adapter                                                 #
# --------------------------------------------------------------------------- #


class ControlNetAdapter:
    """Profiles one StableDiffusionControlNet sampling call (full denoise loop).

    Latency is dominated by ``num_inference_steps``. Variants here are precision
    levels (fp32 / fp16 / bf16) since the whole pipeline is not TorchScript-able
    and already runs under no_grad internally.
    """

    PROMPT = "a realistic color photo"

    def __init__(
        self,
        device: torch.device,
        resolution: int,
        cond_type: str,
        base_model_id: str,
        controlnet_id: str,
        steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        hf_cache_dir: str = "",
        local_files_only: bool = False,
    ) -> None:
        if device.type != "cuda":
            print(
                "[warn] ControlNet profiling on CPU is extremely slow; "
                "a CUDA device is strongly recommended."
            )
        self.name = "controlnet"
        self.device = device
        self.resolution = resolution
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self._current_dtype: torch.dtype | None = None

        self.pipe = self._build_pipe(
            cond_type=cond_type,
            base_model_id=base_model_id,
            controlnet_id=controlnet_id,
            hf_cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )

    def _build_pipe(
        self,
        cond_type: str,
        base_model_id: str,
        controlnet_id: str,
        hf_cache_dir: str,
        local_files_only: bool,
    ):
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "diffusers/transformers required for ControlNet profiling. "
                "Install via requirements-controlnet.txt."
            ) from exc

        # Reuse the training script's cond-type -> ControlNet checkpoint mapping.
        from controlnet_recolor_places365 import COND_TO_CONTROLNET

        cn_id = controlnet_id or COND_TO_CONTROLNET.get(cond_type)
        if cn_id is None:
            raise ValueError(
                f"cond-type '{cond_type}' has no pretrained ControlNet head; "
                "pass --controlnet-id explicitly."
            )

        hf_kwargs = {
            "cache_dir": hf_cache_dir or None,
            "local_files_only": local_files_only,
        }
        print(f"Loading ControlNet head: {cn_id}")
        controlnet = ControlNetModel.from_pretrained(cn_id, **hf_kwargs)
        print(f"Loading base pipeline: {base_model_id}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            **hf_kwargs,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def param_count(self) -> int:
        total = 0
        for attr in ("unet", "vae", "text_encoder", "controlnet"):
            comp = getattr(self.pipe, attr, None)
            if comp is not None and hasattr(comp, "parameters"):
                total += _count_params(comp)
        return total

    def default_variants(self) -> List[str]:
        if self.device.type == "cuda":
            return ["fp32", "fp16", "bf16"]
        return ["fp32"]

    def make_inputs(self, batch_size: int) -> List["Image.Image"]:  # noqa: F821
        from PIL import Image

        imgs = []
        for _ in range(batch_size):
            arr = np.random.randint(
                0, 256, (self.resolution, self.resolution, 3), dtype=np.uint8
            )
            imgs.append(Image.fromarray(arr))
        return imgs

    def _set_dtype(self, dtype: torch.dtype) -> None:
        if self._current_dtype is not dtype:
            self.pipe = self.pipe.to(device=self.device, dtype=dtype)
            self._current_dtype = dtype

    def prepare(self, variant: str) -> Callable[[Any], Any]:
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(variant)
        if dtype is None:
            raise ValueError(f"Unknown variant for controlnet: {variant}")
        if dtype is not torch.float32 and self.device.type != "cuda":
            raise ValueError(f"{variant} requires CUDA")
        self._set_dtype(dtype)

        def call(cond_imgs: List[Any]) -> Any:
            with torch.inference_mode():
                return self.pipe(
                    prompt=[self.PROMPT] * len(cond_imgs),
                    image=cond_imgs,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    output_type="pil",
                ).images

        return call


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def build_adapter(args, device: torch.device):
    """Construct the requested adapter from parsed CLI args."""
    if args.model == "cnn":
        from baseline_cnn_places365 import ColorizationCNN

        res = args.resolution or 128
        return TorchModuleAdapter(
            name="cnn",
            model=ColorizationCNN(),
            input_shape=(1, res, res),
            device=device,
        )

    if args.model == "resnet18":
        import torchvision

        res = args.resolution or 224
        model = torchvision.models.resnet18(weights=None)
        return TorchModuleAdapter(
            name="resnet18",
            model=model,
            input_shape=(3, res, res),
            device=device,
        )

    if args.model == "controlnet":
        res = args.resolution or 512
        return ControlNetAdapter(
            device=device,
            resolution=res,
            cond_type=args.cond_type,
            base_model_id=args.base_model_id,
            controlnet_id=args.controlnet_id,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            hf_cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )

    raise ValueError(f"Unknown model: {args.model}")
