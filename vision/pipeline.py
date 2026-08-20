"""Single-pass preparation and reuse for the vision pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

import cv2
import numpy as np

from vision.config import VisionConfig
from vision.maskers.base import TextMasker
from vision.maskers.heuristic import HeuristicTextMasker
from vision.maskers.hybrid import HybridTextMasker
from vision.region_analysis import analyze_region
from vision.types import (
    BBox,
    BubbleInstance,
    ErasabilityDecision,
    EraseMethod,
    EraseResult,
    MaskResult,
    PreparedBlock,
    RegionAnalysis,
)


class VisionPipeline:
    """Prepare each OCR block once, then reuse its analysis and mask."""

    def __init__(
        self,
        masker: TextMasker | None = None,
        bubble_segmenter: object | None = None,
        config: VisionConfig | None = None,
    ) -> None:
        self.config = config or _load_default_config()
        self.masker = masker or build_text_masker(self.config)
        self.bubble_segmenter = bubble_segmenter

    def prepare_page(
        self,
        image: np.ndarray,
        blocks: Iterable[Mapping[str, object]],
    ) -> list[PreparedBlock]:
        bubbles = self._segment_page(image)
        prepared: list[PreparedBlock] = []
        for index, block in enumerate(blocks):
            bbox = _normalize_bbox(block.get("bbox"), image.shape[:2])
            text = str(block.get("text", ""))
            region = analyze_region(image, bbox)
            if region is None:
                raise ValueError(f"block {index} has an empty bbox")
            bubble = _match_bubble(bbox, bubbles)
            mask_result = self.masker.generate(image, bbox, text, region, bubble)
            decision = score_erasability(region, mask_result, text)
            prepared.append(
                PreparedBlock(
                    block_id=_block_id(index, bbox),
                    text=text,
                    bbox=bbox,
                    region=region,
                    mask_result=mask_result,
                    decision=decision,
                    erase_method=_choose_erase_method(
                        region, mask_result, decision, self.config
                    ),
                )
            )
        return prepared

    def assess(self, prepared: PreparedBlock) -> ErasabilityDecision:
        return prepared.decision

    def erase_block(
        self, image: np.ndarray, prepared: PreparedBlock
    ) -> EraseResult:
        return erase_prepared_block(image, prepared)

    def _segment_page(self, image: np.ndarray) -> list[BubbleInstance]:
        if self.bubble_segmenter is None:
            return []
        segment = getattr(self.bubble_segmenter, "segment", None)
        if segment is None:
            raise TypeError("bubble_segmenter must provide segment(image)")
        return list(segment(image))


def score_erasability(
    region: RegionAnalysis,
    mask_result: MaskResult,
    text: str,
) -> ErasabilityDecision:
    """Score a prepared mask conservatively without regenerating it."""
    coverage = mask_result.coverage
    if coverage <= 0:
        return ErasabilityDecision(False, "no_text_mask", 0.0, False)
    if coverage > 0.65:
        return ErasabilityDecision(False, "excessive_mask", 0.25, True)
    if (
        region.uniformity == "complex"
        and region.texture_std > 55
        and region.bubble_context != "in_bubble"
    ):
        return ErasabilityDecision(False, "complex_artwork", 0.30, True)
    if mask_result.edge_touch_ratio > 0.28 and region.bubble_context != "in_bubble":
        return ErasabilityDecision(False, "risky_background", 0.35, True)

    score = 0.15
    if region.bubble_context == "in_bubble":
        score += 0.45
    if region.uniformity == "uniform":
        score += 0.30
    elif region.uniformity == "textured" and region.intensity_std < 28:
        score += 0.15
    if region.intensity_std < 12:
        score += 0.15
    if region.texture_std < 28:
        score += 0.15
    if region.edge_score < 25:
        score += 0.15
    score = max(0.0, min(1.0, score))
    safe = score >= 0.55
    if region.bubble_context == "in_bubble":
        reason = "in_bubble"
    elif region.uniformity == "uniform":
        reason = "uniform_background"
    elif safe:
        reason = "text_like_mask"
    else:
        reason = "risky_background"
    return ErasabilityDecision(safe, reason, score, not safe)


def build_text_masker(config: VisionConfig) -> TextMasker:
    """Select a masker explicitly, keeping gated auto rollout fail-closed."""
    backend = config.mask_backend
    if backend == "heuristic":
        return HeuristicTextMasker(config.text_mask)
    if backend == "hybrid" or (backend == "auto" and config.hybrid_gate_passed):
        return HybridTextMasker(
            config.text_mask, bubble_border_px=config.bubble.safe_border_px
        )
    if backend == "auto":
        return HeuristicTextMasker(config.text_mask)
    raise RuntimeError("neural text masking backend is not installed yet")


def erase_prepared_block(
    image: np.ndarray,
    prepared: PreparedBlock,
) -> EraseResult:
    """Apply the already prepared mask to ``image`` without inference."""
    started = perf_counter()
    method = prepared.erase_method
    if method == "preserve" or not np.any(prepared.mask_result.mask):
        return EraseResult(
            method="preserve",
            changed_pixels=0,
            elapsed_ms=(perf_counter() - started) * 1000,
            warning=None,
        )

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = prepared.mask_result.roi_bbox
    full_mask[y1:y2, x1:x2] = prepared.mask_result.mask
    before = image[full_mask > 0].copy()
    warning = None

    if method == "flat":
        image[full_mask > 0] = prepared.region.mean_bgr
    else:
        restored = cv2.inpaint(image, full_mask, 3, cv2.INPAINT_TELEA)
        image[full_mask > 0] = restored[full_mask > 0]
        if method == "lama_full_page":
            warning = "LaMa runtime is not available yet; used Telea compatibility fallback"

    after = image[full_mask > 0]
    changed_pixels = int(np.count_nonzero(np.any(before != after, axis=1)))
    return EraseResult(
        method=method,
        changed_pixels=changed_pixels,
        elapsed_ms=(perf_counter() - started) * 1000,
        warning=warning,
    )


def _choose_erase_method(
    region: RegionAnalysis,
    mask_result: MaskResult,
    decision: ErasabilityDecision,
    config: VisionConfig,
) -> EraseMethod:
    if not decision.safe:
        return "preserve"
    if (
        region.uniformity == "uniform"
        and region.texture_std <= config.inpaint.flat_max_texture_std
    ):
        return "flat"
    if region.bubble_context == "in_bubble":
        return "telea"
    if region.uniformity != "complex":
        return "telea"
    return "lama_full_page"


def _normalize_bbox(value: object, image_shape: tuple[int, int]) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise ValueError("block bbox must contain four coordinates")
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError) as exc:
        raise ValueError("block bbox contains a non-numeric coordinate") from exc
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    image_height, image_width = image_shape
    bbox = (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("block bbox does not intersect the image")
    return bbox


def _block_id(index: int, bbox: BBox) -> str:
    return f"block-{index:04d}-{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}"


def _match_bubble(
    bbox: BBox, bubbles: list[BubbleInstance]
) -> BubbleInstance | None:
    if not bubbles:
        return None
    best = max(bubbles, key=lambda bubble: _intersection_over_union(bbox, bubble.bbox))
    return best if _intersection_over_union(bbox, best.bbox) > 0 else None


def _intersection_over_union(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def _load_default_config() -> VisionConfig:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "vision.json"
    return VisionConfig.load(config_path)
