"""Full-page-first LaMa orchestration with a context-preserving OOM path."""

from __future__ import annotations

from dataclasses import dataclass
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
            self.backend.clear_cuda_cache()
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

        output = image.copy()
        crop_output = _composite(crop_image, restored_crop, crop_mask)
        target = output[y1:y2, x1:x2]
        target[crop_mask > 0] = crop_output[crop_mask > 0]
        self.last_run = LamaRunStats(
            "context_crop", 2, full_shape, crop_image.shape[:2]
        )
        return output


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
