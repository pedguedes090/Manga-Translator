"""Shared types for vision analysis, masking, and inpainting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


BBox = tuple[int, int, int, int]
EraseMethod = Literal["preserve", "flat", "telea", "lama_full_page"]


@dataclass(frozen=True)
class RegionAnalysis:
    mean_bgr: tuple[int, int, int]
    mean_intensity: float
    intensity_std: float
    edge_score: float
    texture_std: float
    dominant_tone: Literal["dark", "light", "mid"]
    uniformity: Literal["uniform", "textured", "complex"]
    bubble_context: Literal[
        "in_bubble",
        "on_artwork_dark",
        "on_artwork_light",
        "on_artwork_mixed",
    ]


@dataclass(frozen=True)
class BubbleInstance:
    instance_id: str
    bbox: BBox
    mask: np.ndarray
    confidence: float
    category: Literal["speech_bubble", "thought_bubble", "caption_box"]


@dataclass
class MaskResult:
    roi_bbox: BBox
    mask: np.ndarray
    probability: np.ndarray | None
    bubble_mask: np.ndarray | None
    coverage: float
    confidence: float
    edge_touch_ratio: float
    backend: str
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErasabilityDecision:
    safe: bool
    reason: str
    score: float
    requires_review: bool


@dataclass
class PreparedBlock:
    block_id: str
    text: str
    bbox: BBox
    region: RegionAnalysis
    mask_result: MaskResult
    decision: ErasabilityDecision
    erase_method: EraseMethod
    mask_ref: Path | None = None

    def to_summary(self) -> dict[str, object]:
        """Return JSON-safe metadata without runtime NumPy arrays."""
        return {
            "block_id": self.block_id,
            "text": self.text,
            "bbox": list(self.bbox),
            "mask_ref": self.mask_ref.as_posix() if self.mask_ref else None,
            "decision": {
                "safe": self.decision.safe,
                "reason": self.decision.reason,
                "score": self.decision.score,
                "requires_review": self.decision.requires_review,
            },
            "backend": self.mask_result.backend,
            "coverage": self.mask_result.coverage,
            "confidence": self.mask_result.confidence,
            "erase_method": self.erase_method,
        }


@dataclass(frozen=True)
class EraseResult:
    method: EraseMethod
    changed_pixels: int
    elapsed_ms: float
    warning: str | None
