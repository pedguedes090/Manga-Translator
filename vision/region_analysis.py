"""Typed, non-mutating analysis of OCR image regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.types import BBox, RegionAnalysis


@dataclass(frozen=True)
class _RegionDiagnostics:
    mean_bgr: np.ndarray
    border_interior_contrast: float
    is_bubble: bool


def analyze_region(image: np.ndarray, bbox: BBox) -> RegionAnalysis | None:
    """Analyze background and bubble characteristics without changing ``image``."""
    analysis, _ = _analyze_region_with_diagnostics(image, bbox)
    return analysis


def _analyze_region_with_diagnostics(
    image: np.ndarray, bbox: BBox
) -> tuple[RegionAnalysis | None, _RegionDiagnostics | None]:
    x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
    image_height, image_width = image.shape[:2]
    x2 = min(x2, image_width)
    y2 = min(y2, image_height)
    if x2 <= x1 or y2 <= y1:
        return None, None

    height, width = y2 - y1, x2 - x1
    margin = max(2, min(height, width) // 8)
    border_samples: list[np.ndarray] = []

    border_y1 = max(0, y1 - margin)
    if border_y1 < y1:
        border_samples.append(image[border_y1:y1, x1:x2])
    border_y2 = min(image_height, y2 + margin)
    if y2 < border_y2:
        border_samples.append(image[y2:border_y2, x1:x2])
    border_x1 = max(0, x1 - margin)
    if border_x1 < x1:
        border_samples.append(image[y1:y2, border_x1:x1])
    border_x2 = min(image_width, x2 + margin)
    if x2 < border_x2:
        border_samples.append(image[y1:y2, x2:border_x2])

    ring_thickness = max(2, min(height, width) // 12)
    interior_samples: list[np.ndarray] = []
    top_y2 = min(y1 + ring_thickness, y2)
    interior_samples.append(image[y1:top_y2, x1:x2])
    bottom_y1 = max(y1, y2 - ring_thickness)
    interior_samples.append(image[bottom_y1:y2, x1:x2])
    left_x2 = min(x1 + ring_thickness, x2)
    if top_y2 < bottom_y1:
        interior_samples.append(image[top_y2:bottom_y1, x1:left_x2])
    right_x1 = max(x1, x2 - ring_thickness)
    if top_y2 < bottom_y1:
        interior_samples.append(image[top_y2:bottom_y1, right_x1:x2])

    edge_thickness = max(1, min(height, width) // 30)
    edge_samples: list[np.ndarray] = []
    edge_y2 = min(y1 + edge_thickness, y2)
    edge_samples.append(image[y1:edge_y2, x1:x2])
    edge_bottom_y1 = max(y1, y2 - edge_thickness)
    edge_samples.append(image[edge_bottom_y1:y2, x1:x2])
    edge_left_x2 = min(x1 + edge_thickness, x2)
    if edge_y2 < edge_bottom_y1:
        edge_samples.append(image[edge_y2:edge_bottom_y1, x1:edge_left_x2])
    edge_right_x1 = max(x1, x2 - edge_thickness)
    if edge_y2 < edge_bottom_y1:
        edge_samples.append(image[edge_y2:edge_bottom_y1, edge_right_x1:x2])

    border_pixels = _combine_samples(border_samples)
    interior_pixels = _combine_samples(interior_samples)
    edge_pixels = _combine_samples(edge_samples)
    region_pixels = image[y1:y2, x1:x2].reshape(-1, 3)

    if interior_pixels is not None and len(interior_pixels) >= 10:
        median_bgr = np.median(interior_pixels, axis=0).astype(np.float32)
    elif edge_pixels is not None and len(edge_pixels) >= 10:
        median_bgr = np.median(edge_pixels, axis=0).astype(np.float32)
    else:
        median_bgr = np.median(region_pixels, axis=0).astype(np.float32)

    mean_intensity = _luminance(median_bgr)
    if interior_pixels is not None and len(interior_pixels) >= 20:
        background_gray = _to_gray_values(interior_pixels)
    else:
        background_gray = _to_gray_values(region_pixels)
    median_gray = np.median(background_gray)
    mad = float(np.median(np.abs(background_gray - median_gray)))
    intensity_std = mad * 1.4826
    texture_std = float(np.std(background_gray))

    if height >= 17 and width >= 17:
        gray_patch = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        scale = max(1, min(width, height) // 80)
        small = gray_patch[::scale, ::scale]
        gradient_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        edge_score = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))
    else:
        edge_score = 0.0

    if border_pixels is not None and len(border_pixels) >= 10:
        border_mean = np.mean(border_pixels, axis=0).astype(np.float32)
        border_intensity = _luminance(border_mean)
    else:
        border_intensity = mean_intensity

    if edge_pixels is not None and len(edge_pixels) >= 10:
        edge_mean = np.mean(edge_pixels, axis=0).astype(np.float32)
        edge_intensity = _luminance(edge_mean)
        edge_std = float(np.mean(np.std(edge_pixels, axis=0)))
    else:
        edge_intensity = mean_intensity
        edge_std = intensity_std

    border_interior_contrast = abs(border_intensity - mean_intensity)
    edge_darker_than_interior = (mean_intensity - edge_intensity) > 25
    edge_lighter_than_interior = (edge_intensity - mean_intensity) > 25
    edge_has_strong_outline = (
        edge_std > 25 and abs(edge_intensity - mean_intensity) > 15
    )

    bubble_confidence = 0.0
    if border_interior_contrast > 40:
        bubble_confidence += 0.35
    if border_interior_contrast > 20:
        bubble_confidence += 0.15
    if edge_darker_than_interior or edge_lighter_than_interior:
        bubble_confidence += 0.25
    if edge_has_strong_outline:
        bubble_confidence += 0.20
    if intensity_std < 20:
        bubble_confidence += 0.20
    elif intensity_std < 35:
        bubble_confidence += 0.10
    if edge_score < 20:
        bubble_confidence += 0.15
    elif edge_score > 45:
        bubble_confidence -= 0.15
    is_bubble = bubble_confidence >= 0.55

    if is_bubble:
        bubble_context = "in_bubble"
    elif mean_intensity < 90:
        bubble_context = "on_artwork_dark"
    elif mean_intensity > 170:
        bubble_context = "on_artwork_light"
    else:
        bubble_context = "on_artwork_mixed"

    if intensity_std < 18:
        uniformity = "uniform"
    elif intensity_std < 50 and edge_score < 50:
        uniformity = "textured"
    else:
        uniformity = "complex"

    if mean_intensity < 80:
        dominant_tone = "dark"
    elif mean_intensity > 170:
        dominant_tone = "light"
    else:
        dominant_tone = "mid"

    analysis = RegionAnalysis(
        mean_bgr=tuple(int(round(float(channel))) for channel in median_bgr),
        mean_intensity=mean_intensity,
        intensity_std=float(intensity_std),
        edge_score=edge_score,
        texture_std=texture_std,
        dominant_tone=dominant_tone,
        uniformity=uniformity,
        bubble_context=bubble_context,
    )
    diagnostics = _RegionDiagnostics(
        mean_bgr=median_bgr,
        border_interior_contrast=float(border_interior_contrast),
        is_bubble=is_bubble,
    )
    return analysis, diagnostics


def _combine_samples(samples: list[np.ndarray]) -> np.ndarray | None:
    if not samples:
        return None
    return np.concatenate([sample.reshape(-1, 3) for sample in samples], axis=0)


def _to_gray_values(pixels: np.ndarray) -> np.ndarray:
    return (
        0.114 * pixels[:, 0].astype(np.float64)
        + 0.587 * pixels[:, 1].astype(np.float64)
        + 0.299 * pixels[:, 2].astype(np.float64)
    )


def _luminance(bgr: np.ndarray) -> float:
    return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
