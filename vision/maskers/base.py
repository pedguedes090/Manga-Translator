"""Interface shared by text-mask backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vision.types import BBox, BubbleInstance, MaskResult, RegionAnalysis


class TextMasker(ABC):
    @abstractmethod
    def generate(
        self,
        image: np.ndarray,
        bbox: BBox,
        text: str,
        region: RegionAnalysis,
        bubble: BubbleInstance | None,
    ) -> MaskResult:
        """Generate a mask local to ``MaskResult.roi_bbox``."""
        raise NotImplementedError
