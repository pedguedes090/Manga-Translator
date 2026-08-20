"""Reusable binary-mask postprocessing primitives."""

from __future__ import annotations

import cv2
import numpy as np


def hysteresis_mask(probability: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak pixels only when connected to a strong seed."""
    if probability.ndim != 2:
        raise ValueError("probability must be a two-dimensional array")
    if not 0 <= low < high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")

    weak = (probability >= low).astype(np.uint8)
    strong = probability >= high
    label_count, labels = cv2.connectedComponents(weak, connectivity=8)
    output = np.zeros(probability.shape, dtype=np.uint8)
    for label in range(1, label_count):
        component = labels == label
        if np.any(strong & component):
            output[component] = 255
    return output


def keep_ocr_anchored_components(
    mask: np.ndarray,
    inner_rect: tuple[int, int, int, int],
    min_overlap: float = 0.30,
) -> np.ndarray:
    """Keep connected components sufficiently anchored in the OCR bbox."""
    if not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must be between zero and one")
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")

    height, width = mask.shape
    x1, y1, x2, y2 = inner_rect
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    inner = np.zeros(mask.shape, dtype=bool)
    inner[y1:y2, x1:x2] = True

    binary = (mask > 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        overlap = np.count_nonzero(component & inner) / float(area)
        if overlap >= min_overlap:
            output[component] = 255
    return output


def apply_bubble_gate(
    mask: np.ndarray,
    bubble_mask: np.ndarray,
    border_px: int,
) -> np.ndarray:
    """Restrict a text mask to the safe interior of a bubble mask."""
    if mask.shape != bubble_mask.shape:
        raise ValueError("mask and bubble_mask must have the same shape")
    if border_px < 0:
        raise ValueError("border_px cannot be negative")

    safe_interior = (bubble_mask > 0).astype(np.uint8) * 255
    if border_px:
        kernel_size = border_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        safe_interior = cv2.erode(safe_interior, kernel, iterations=1)
    return cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, safe_interior)
