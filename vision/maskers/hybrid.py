"""Fast hybrid text masker using local-background residuals."""

from __future__ import annotations

import cv2
import numpy as np

from vision.config import TextMaskConfig
from vision.maskers.base import TextMasker
from vision.maskers.heuristic import remove_screentone_dots
from vision.postprocess import (
    apply_bubble_gate,
    hysteresis_mask,
    keep_ocr_anchored_components,
)
from vision.types import BBox, BubbleInstance, MaskResult, RegionAnalysis


_DEFAULT_CONFIG = TextMaskConfig(
    input_size=512,
    crop_padding_ratio=0.12,
    prob_high=0.62,
    prob_low=0.34,
    max_coverage=0.65,
    max_bubble_border_overlap=0.02,
    dilation_min_px=1,
    dilation_max_px=4,
)


class HybridTextMasker(TextMasker):
    """Segment strokes from local contrast without a neural model."""

    def __init__(
        self,
        config: TextMaskConfig | None = None,
        bubble_border_px: int = 3,
    ) -> None:
        self.config = config or _DEFAULT_CONFIG
        if bubble_border_px < 0:
            raise ValueError("bubble_border_px cannot be negative")
        self.bubble_border_px = bubble_border_px

    def generate(
        self,
        image: np.ndarray,
        bbox: BBox,
        text: str,
        region: RegionAnalysis,
        bubble: BubbleInstance | None,
    ) -> MaskResult:
        roi_bbox, inner_rect = _expanded_roi(
            bbox, image.shape[:2], self.config.crop_padding_ratio
        )
        x1, y1, x2, y2 = roi_bbox
        roi = image[y1:y2, x1:x2]
        probability = _local_background_residual(roi)
        raw_mask = hysteresis_mask(
            probability, self.config.prob_low, self.config.prob_high
        )
        raw_components = _component_count(raw_mask)
        mask = keep_ocr_anchored_components(raw_mask, inner_rect)
        anchored_components = _component_count(mask)
        filtered = remove_screentone_dots(mask)
        if filtered is not None:
            mask = filtered

        bubble_mask = _bubble_mask_for_roi(bubble, roi_bbox, image.shape[:2])
        if bubble_mask is not None:
            mask = apply_bubble_gate(
                mask, bubble_mask, border_px=self.bubble_border_px
            )

        dilation_radius = _estimate_dilation_radius(
            mask,
            self.config.dilation_min_px,
            self.config.dilation_max_px,
        )
        if np.any(mask):
            kernel_size = dilation_radius * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
            if bubble_mask is not None:
                mask = apply_bubble_gate(
                    mask, bubble_mask, border_px=self.bubble_border_px
                )
            _clear_crop_border(mask, dilation_radius)

        nonzero = int(np.count_nonzero(mask))
        coverage = nonzero / float(max(mask.size, 1))
        confidence = float(np.mean(probability[mask > 0])) if nonzero else 0.0
        return MaskResult(
            roi_bbox=roi_bbox,
            mask=mask,
            probability=probability,
            bubble_mask=bubble_mask,
            coverage=float(coverage),
            confidence=confidence,
            edge_touch_ratio=_edge_touch_ratio(mask),
            backend="hybrid",
            debug={
                "raw_residual": probability,
                "component_counts": {
                    "raw": raw_components,
                    "anchored": anchored_components,
                    "final": _component_count(mask),
                },
                "dilation_radius": dilation_radius,
            },
        )


def _expanded_roi(
    bbox: BBox,
    image_shape: tuple[int, int],
    padding_ratio: float,
) -> tuple[BBox, tuple[int, int, int, int]]:
    image_height, image_width = image_shape
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must contain a non-empty image region")
    padding_x = max(1, int(round((x2 - x1) * padding_ratio)))
    padding_y = max(1, int(round((y2 - y1) * padding_ratio)))
    roi_x1 = max(0, x1 - padding_x)
    roi_y1 = max(0, y1 - padding_y)
    roi_x2 = min(image_width, x2 + padding_x)
    roi_y2 = min(image_height, y2 + padding_y)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        raise ValueError("bbox does not intersect the image")
    return (
        (roi_x1, roi_y1, roi_x2, roi_y2),
        (x1 - roi_x1, y1 - roi_y1, x2 - roi_x1, y2 - roi_y1),
    )


def _local_background_residual(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    kernel_size = _median_kernel_size(min(gray.shape))
    background_gray = cv2.medianBlur(gray, kernel_size)
    background_lab = np.stack(
        [cv2.medianBlur(lab[:, :, channel], kernel_size) for channel in range(3)],
        axis=2,
    )

    gray_residual = cv2.absdiff(gray, background_gray).astype(np.float32) / 255.0
    lab_delta = lab.astype(np.float32) - background_lab.astype(np.float32)
    lab_residual = np.linalg.norm(lab_delta, axis=2) / (np.sqrt(3.0) * 255.0)
    return np.clip(np.maximum(gray_residual, lab_residual), 0.0, 1.0).astype(
        np.float32
    )


def _median_kernel_size(short_side: int) -> int:
    if short_side < 3:
        raise ValueError("ROI is too small for hybrid masking")
    return min(9, short_side if short_side % 2 else short_side - 1)


def _estimate_dilation_radius(mask: np.ndarray, minimum: int, maximum: int) -> int:
    if not np.any(mask):
        return minimum
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    stroke_radius = float(np.median(distance[distance > 0]))
    return max(minimum, min(maximum, int(round(stroke_radius))))


def _bubble_mask_for_roi(
    bubble: BubbleInstance | None,
    roi_bbox: BBox,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if bubble is None:
        return None
    if bubble.mask.shape == image_shape:
        x1, y1, x2, y2 = roi_bbox
        return bubble.mask[y1:y2, x1:x2]
    expected_shape = (roi_bbox[3] - roi_bbox[1], roi_bbox[2] - roi_bbox[0])
    if bubble.mask.shape != expected_shape:
        raise ValueError("bubble mask must be full-page or ROI-local")
    return bubble.mask


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), 8)
    return int(count - 1)


def _clear_crop_border(mask: np.ndarray, radius: int) -> None:
    mask[:radius, :] = 0
    mask[-radius:, :] = 0
    mask[:, :radius] = 0
    mask[:, -radius:] = 0


def _edge_touch_ratio(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    edge = np.zeros(mask.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return float(np.count_nonzero((mask > 0) & edge) / max(mask.size, 1))
