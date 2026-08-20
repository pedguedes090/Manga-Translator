"""OpenCV text masker extracted from the legacy rendering path."""

from __future__ import annotations

import cv2
import numpy as np

from vision.config import TextMaskConfig
from vision.maskers.base import TextMasker
from vision.types import BBox, BubbleInstance, MaskResult, RegionAnalysis


class HeuristicTextMasker(TextMasker):
    """Fast CPU baseline that preserves the existing stroke-mask behavior."""

    def __init__(self, config: TextMaskConfig | None = None) -> None:
        self.config = config

    def generate(
        self,
        image: np.ndarray,
        bbox: BBox,
        text: str,
        region: RegionAnalysis,
        bubble: BubbleInstance | None,
    ) -> MaskResult:
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        roi_bbox = (
            max(0, min(image_width, int(x1))),
            max(0, min(image_height, int(y1))),
            max(0, min(image_width, int(x2))),
            max(0, min(image_height, int(y2))),
        )
        x1, y1, x2, y2 = roi_bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must contain a non-empty image region")

        appearance = {
            "intensity_std": region.intensity_std,
            "uniformity": region.uniformity,
        }
        mask = build_text_stroke_mask(
            image[y1:y2, x1:x2], region.mean_bgr, appearance
        )
        if mask is None:
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

        nonzero = int(np.count_nonzero(mask))
        coverage = nonzero / float(max(mask.size, 1))
        edge_touch_ratio = edge_touching_component_coverage(mask)
        bubble_mask = _bubble_mask_for_roi(bubble, roi_bbox, image.shape[:2])
        return MaskResult(
            roi_bbox=roi_bbox,
            mask=mask,
            probability=None,
            bubble_mask=bubble_mask,
            coverage=float(coverage),
            confidence=1.0 if nonzero else 0.0,
            edge_touch_ratio=float(edge_touch_ratio),
            backend="heuristic",
            debug={},
        )


def build_text_stroke_mask(
    roi: np.ndarray | None,
    fill_color: tuple[int, int, int] | np.ndarray,
    appearance: dict[str, object],
) -> np.ndarray | None:
    """Detect only the original ink strokes inside an OCR region."""
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    fill_gray = _luminance_bgr(fill_color)
    intensity_std = float(appearance.get("intensity_std", 0) or 0)
    threshold_gap = max(14, min(42, int(18 + intensity_std * 0.25)))

    roi_float = roi.astype(np.float32)
    fill = np.array(fill_color, dtype=np.float32)
    color_dist = np.sqrt(np.sum((roi_float - fill) ** 2, axis=2))
    preserve_uniform_bright_halo = False

    if fill_gray >= 150:
        dark_strokes = gray < (fill_gray - threshold_gap)
        color_outliers = (color_dist > 45) & (
            gray < fill_gray - max(10, threshold_gap // 2)
        )
        mask = dark_strokes | color_outliers
        if fill_gray < 235:
            bright_halo = (gray > fill_gray + max(18, threshold_gap // 2)) & (
                color_dist > 28
            )
            mask |= bright_halo
            dark_coverage = np.count_nonzero(dark_strokes) / max(mask.size, 1)
            halo_coverage = np.count_nonzero(bright_halo) / max(mask.size, 1)
            preserve_uniform_bright_halo = (
                appearance.get("uniformity") == "uniform"
                and intensity_std < 10
                and dark_coverage > 0.04
                and halo_coverage > 0.04
            )

        base_coverage = np.count_nonzero(mask) / max(mask.size, 1)
        if base_coverage < 0.12 and min(gray.shape[:2]) >= 15:
            block_size = max(15, (min(gray.shape[:2]) // 3) | 1)
            block_size = min(block_size, 45)
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                9,
            ) > 0
            mask |= (
                adaptive
                & (gray < fill_gray - max(14, threshold_gap // 2))
                & (color_dist > 20)
            )
    elif fill_gray <= 105:
        light_strokes = gray > (fill_gray + threshold_gap)
        color_outliers = (color_dist > 38) & (gray > fill_gray - 8)
        mask = light_strokes | color_outliers
    else:
        mask = color_dist > 45

    mask = mask.astype(np.uint8) * 255
    coverage = np.count_nonzero(mask) / max(mask.size, 1)
    if coverage > 0.42 and not preserve_uniform_bright_halo:
        if fill_gray >= 150:
            mask = (gray < fill_gray - max(26, threshold_gap + 8)).astype(np.uint8) * 255
        elif fill_gray <= 105:
            mask = (gray > fill_gray + max(26, threshold_gap + 8)).astype(np.uint8) * 255
        else:
            mask = (color_dist > 65).astype(np.uint8) * 255

    kernel = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    mask = filter_text_mask_components(mask)
    mask_coverage = np.count_nonzero(mask) / max(mask.size, 1)
    if not (preserve_uniform_bright_halo and mask_coverage > 0.64):
        mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_coverage = np.count_nonzero(mask) / max(mask.size, 1)
        if (
            preserve_uniform_bright_halo
            and mask_coverage > 0.50
            and dilated_coverage < 0.66
        ):
            mask = cv2.dilate(mask, kernel, iterations=1)

    final_coverage = np.count_nonzero(mask) / max(mask.size, 1)
    if final_coverage > 0.38 and not preserve_uniform_bright_halo:
        if fill_gray >= 150:
            mask = (gray < fill_gray - max(28, threshold_gap + 8)).astype(np.uint8) * 255
        elif fill_gray <= 105:
            mask = (gray > fill_gray + max(28, threshold_gap + 8)).astype(np.uint8) * 255
        else:
            mask = (color_dist > 70).astype(np.uint8) * 255
        mask = filter_text_mask_components(mask)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def filter_text_mask_components(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return mask

    height, width = mask.shape[:2]
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    max_area = max(12, int(height * width * 0.65))
    for label in range(1, label_count):
        _, _, component_width, component_height, area = stats[label]
        if area < 2 or area > max_area:
            continue
        if component_width > width * 0.96 and component_height > height * 0.55:
            continue
        filtered[labels == label] = 255
    return filtered


def filter_components_outside_inner(
    mask: np.ndarray | None,
    inner_rect: tuple[int, int, int, int] | None,
    min_overlap: float = 0.30,
) -> np.ndarray | None:
    """Keep components anchored inside the unpadded OCR rectangle."""
    if mask is None or inner_rect is None:
        return mask
    inner_x1, inner_y1, inner_x2, inner_y2 = inner_rect
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    inner = np.zeros_like(mask, dtype=bool)
    inner[inner_y1:inner_y2, inner_x1:inner_x2] = True
    filtered = np.zeros_like(mask)
    mask_height, mask_width = mask.shape[:2]

    for label in range(1, label_count):
        x, y, component_width, component_height, area = stats[label]
        if area <= 0:
            continue
        touches_roi_edge = (
            x <= 1
            or y <= 1
            or x + component_width >= mask_width - 1
            or y + component_height >= mask_height - 1
        )
        component = labels == label
        overlap = np.count_nonzero(component & inner) / float(area)
        touches_left_or_right = (
            x <= 1 or x + component_width >= mask_width - 1
        )
        touches_top_or_bottom = (
            y <= 1 or y + component_height >= mask_height - 1
        )
        thin_vertical_border = (
            touches_left_or_right
            and component_width <= max(4, mask_width * 0.06)
            and component_height >= mask_height * 0.45
        )
        thin_horizontal_border = (
            touches_top_or_bottom
            and component_height <= max(4, mask_height * 0.06)
            and component_width >= mask_width * 0.45
        )
        likely_border = thin_vertical_border or thin_horizontal_border
        likely_text_touching_edge = (
            touches_roi_edge
            and overlap >= 0.85
            and not likely_border
            and (
                component_width >= mask_width * 0.08
                or component_height >= mask_height * 0.20
                or area >= mask.size * 0.01
            )
        )
        likely_wide_text_touching_side = (
            touches_roi_edge
            and overlap >= 0.85
            and component_width >= mask_width * 0.35
            and component_height >= mask_height * 0.20
        )
        if (
            touches_roi_edge
            and not likely_text_touching_edge
            and not likely_wide_text_touching_side
        ):
            continue
        if overlap >= min_overlap:
            filtered[component] = 255
    return filtered


def remove_screentone_dots(
    mask: np.ndarray | None,
    tiny_area: int = 30,
    min_count: int = 40,
    min_ratio: float = 0.6,
) -> np.ndarray | None:
    """Remove grids of uniformly tiny halftone components."""
    if mask is None or np.count_nonzero(mask) == 0:
        return mask
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if label_count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    tiny_labels = [index + 1 for index, area in enumerate(areas) if area <= tiny_area]
    total = len(areas)
    if len(tiny_labels) >= min_count and len(tiny_labels) / total >= min_ratio:
        filtered = mask.copy()
        filtered[np.isin(labels, tiny_labels)] = 0
        return filtered
    return mask


def edge_touching_component_coverage(mask: np.ndarray | None) -> float:
    if mask is None or np.count_nonzero(mask) == 0:
        return 0.0
    label_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if label_count <= 1:
        return 0.0

    edge_pixels = 0
    height, width = mask.shape[:2]
    for label in range(1, label_count):
        x, y, component_width, component_height, area = stats[label]
        if (
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        ):
            edge_pixels += int(area)
    return edge_pixels / float(max(mask.size, 1))


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
    return bubble.mask


def _luminance_bgr(color: tuple[int, int, int] | np.ndarray) -> float:
    return float(0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2])
