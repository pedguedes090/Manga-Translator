"""Full-page-first LaMa orchestration with a context-preserving OOM path."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np


class LamaCudaOutOfMemory(RuntimeError):
    """Normalized CUDA OOM raised by a concrete LaMa backend."""


class LamaBackend(Protocol):
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def clear_cuda_cache(self) -> None: ...


@dataclass(frozen=True)
class LamaRunStats:
    mode: str
    calls: int
    full_shape: tuple[int, int]
    retry_shape: tuple[int, int] | None = None
    warning: str | None = None


class TorchLamaBackend:
    """Lazy PyTorch backend for the AnimeMangaInpainting LaMa checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: object | None = None,
        device: str = "cuda",
        precision: str = "fp32",
    ) -> None:
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else None
        )
        self.model = model
        self.device_name = device
        self.precision = precision
        self.last_elapsed_ms = 0.0
        self.last_peak_vram_bytes = 0
        self.last_precision = precision
        self._fp16_healthy = True
        self._torch = None

    def _ensure_model_loaded(self):
        import torch

        self._torch = torch
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the configured LaMa runtime")
        if self.model is None:
            if self.checkpoint_path is None or not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"LaMa checkpoint not found: {self.checkpoint_path}"
                )
            from vision.inpainting.lama_arch import build_lama_model

            self.model = build_lama_model(self.checkpoint_path, device=device)
        eval_model = getattr(self.model, "eval", None)
        if eval_model is not None:
            eval_model()
        return torch, device

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image, mask = _validate_inputs(image, mask)
        torch, device = self._ensure_model_loaded()
        started = perf_counter()
        try:
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            image_tensor = (
                torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            )
            mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))[None]
            image_tensor = image_tensor * (1.0 - mask_tensor)
            height, width = image.shape[:2]
            pad_height = (-height) % 8
            pad_width = (-width) % 8
            if pad_height or pad_width:
                image_tensor = torch.nn.functional.pad(
                    image_tensor,
                    (0, pad_width, 0, pad_height),
                    mode="reflect",
                )
                mask_tensor = torch.nn.functional.pad(
                    mask_tensor,
                    (0, pad_width, 0, pad_height),
                    mode="constant",
                    value=0,
                )
            model_input = torch.cat((image_tensor, mask_tensor), dim=0)
            model_input = model_input[None].to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            use_fp16 = self._fp16_healthy and self._use_fp16(device)
            try:
                output = self._run_model(torch, device, model_input, use_fp16)
            except RuntimeError as exc:
                if not use_fp16 or not _is_fp16_fft_error(exc):
                    raise
                self._fp16_healthy = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                output = self._run_model(torch, device, model_input, False)
                self.last_precision = "fp32_fallback"
            else:
                self.last_precision = "fp16" if use_fp16 else "fp32"
            if use_fp16 and not bool(torch.isfinite(output).all()):
                self._fp16_healthy = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                output = self._run_model(torch, device, model_input, False)
                self.last_precision = "fp32_fallback"
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("LaMa produced non-finite output")
            output = output[0, :, :height, :width].float().cpu().clamp(0, 1)
            result_rgb = (
                output.permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)
            if device.type == "cuda":
                self.last_peak_vram_bytes = int(
                    torch.cuda.max_memory_allocated(device)
                )
            return np.ascontiguousarray(result_rgb[:, :, ::-1])
        except torch.cuda.OutOfMemoryError as exc:
            raise LamaCudaOutOfMemory(str(exc)) from exc
        finally:
            self.last_elapsed_ms = (perf_counter() - started) * 1000.0

    def _use_fp16(self, device: object) -> bool:
        return getattr(device, "type", None) == "cuda" and self.precision == "fp16"

    def _run_model(self, torch, device, model_input, use_fp16: bool):
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            return self.model(model_input)

    def clear_cuda_cache(self) -> None:
        torch = self._torch
        if torch is None:
            try:
                import torch
            except ImportError:
                return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ResilientLamaInpainter:
    """Try one full-resolution page, then a contextual crop only after OOM."""

    def __init__(
        self,
        backend: LamaBackend,
        *,
        context_min_px: int = 256,
        context_max_mask_ratio: float = 0.08,
        telea_radius: int = 3,
    ) -> None:
        if context_min_px < 0:
            raise ValueError("context_min_px must be non-negative")
        if not 0 < context_max_mask_ratio <= 1:
            raise ValueError("context_max_mask_ratio must be in (0, 1]")
        if telea_radius <= 0:
            raise ValueError("telea_radius must be positive")
        self.backend = backend
        self.context_min_px = context_min_px
        self.context_max_mask_ratio = context_max_mask_ratio
        self.telea_radius = telea_radius
        self.last_run: LamaRunStats | None = None

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image, mask = _validate_inputs(image, mask)
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        full_shape = image.shape[:2]
        if not np.any(binary_mask):
            self.last_run = LamaRunStats("empty_mask", 0, full_shape)
            return image.copy()

        try:
            restored = self.backend.inpaint(image.copy(), binary_mask)
            restored = _validate_output(restored, image.shape)
            self.last_run = LamaRunStats("full_page", 1, full_shape)
            return _composite(image, restored, binary_mask)
        except LamaCudaOutOfMemory:
            pass
        finally:
            self.backend.clear_cuda_cache()

        x1, y1, x2, y2 = _context_box(
            binary_mask,
            min_context_px=self.context_min_px,
            max_mask_ratio=self.context_max_mask_ratio,
        )
        crop_image = image[y1:y2, x1:x2].copy()
        crop_mask = binary_mask[y1:y2, x1:x2].copy()
        try:
            restored_crop = self.backend.inpaint(crop_image, crop_mask)
            restored_crop = _validate_output(restored_crop, crop_image.shape)
        except LamaCudaOutOfMemory as exc:
            restored = cv2.inpaint(
                image, binary_mask, self.telea_radius, cv2.INPAINT_TELEA
            )
            warning = f"LaMa CUDA OOM after contextual retry; used Telea: {exc}"
            self.last_run = LamaRunStats(
                "telea_fallback",
                2,
                full_shape,
                crop_image.shape[:2],
                warning,
            )
            return _composite(image, restored, binary_mask)
        finally:
            self.backend.clear_cuda_cache()

        output = image.copy()
        crop_output = _composite(crop_image, restored_crop, crop_mask)
        target = output[y1:y2, x1:x2]
        target[crop_mask > 0] = crop_output[crop_mask > 0]
        self.last_run = LamaRunStats(
            "context_crop", 2, full_shape, crop_image.shape[:2]
        )
        return output


def discover_lama_checkpoint(
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Find a local checkpoint without importing or initializing PyTorch."""
    repository = Path(__file__).resolve().parents[2]
    candidates = [
        explicit_path,
        os.environ.get("LAMA_CHECKPOINT"),
        repository / "models" / "cache" / "lama_large_512px.ckpt",
        repository / "model" / "lama_large_512px.ckpt",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    return None


def build_lama_inpainter(
    checkpoint_path: str | Path | None = None,
    *,
    device: str = "cuda",
    precision: str = "fp32",
    context_min_px: int = 256,
    context_max_mask_ratio: float = 0.08,
    telea_radius: int = 3,
) -> ResilientLamaInpainter | None:
    """Build the lazy production runtime when a local checkpoint is present."""
    checkpoint = discover_lama_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None
    backend = TorchLamaBackend(
        checkpoint,
        device=device,
        precision=precision,
    )
    return ResilientLamaInpainter(
        backend,
        context_min_px=context_min_px,
        context_max_mask_ratio=context_max_mask_ratio,
        telea_radius=telea_radius,
    )


def _validate_inputs(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("LaMa image must be an HxWx3 numpy array")
    if image.dtype != np.uint8:
        raise ValueError("LaMa image must use uint8 pixels")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("LaMa mask must be an HxW numpy array")
    if mask.shape != image.shape[:2]:
        raise ValueError("LaMa mask shape must match the image")
    return image, mask


def _is_fp16_fft_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cufft" in message and "half precision" in message


def _validate_output(output: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(output, np.ndarray) or output.shape != expected_shape:
        raise ValueError("LaMa output must match the input image shape")
    if output.dtype != np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def _composite(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    output = original.copy()
    output[mask > 0] = restored[mask > 0]
    return output


def _context_box(
    mask: np.ndarray,
    *,
    min_context_px: int,
    max_mask_ratio: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    height, width = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    mask_pixels = int(xs.size)

    required_area = min(height * width, int(np.ceil(mask_pixels / max_mask_ratio)))
    pad = min_context_px
    while True:
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(width, x2 + pad)
        bottom = min(height, y2 + pad)
        if (right - left) * (bottom - top) >= required_area:
            return left, top, right, bottom
        if left == 0 and top == 0 and right == width and bottom == height:
            return left, top, right, bottom
        pad = max(pad + 1, int(pad * 1.5))
