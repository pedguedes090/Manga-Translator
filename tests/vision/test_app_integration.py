import numpy as np

from add_text import appearance_for_prepared
from vision.types import (
    ErasabilityDecision,
    MaskResult,
    PreparedBlock,
    RegionAnalysis,
)


def _prepared_block():
    region = RegionAnalysis(
        mean_bgr=(245, 245, 245),
        mean_intensity=245.0,
        intensity_std=2.0,
        edge_score=2.0,
        texture_std=2.0,
        dominant_tone="light",
        uniformity="uniform",
        bubble_context="in_bubble",
    )
    return PreparedBlock(
        block_id="block-0",
        text="TEXT",
        bbox=(2, 2, 8, 8),
        region=region,
        mask_result=MaskResult(
            roi_bbox=(2, 2, 8, 8),
            mask=np.ones((6, 6), dtype=np.uint8) * 255,
            probability=None,
            bubble_mask=None,
            coverage=0.25,
            confidence=0.9,
            edge_touch_ratio=0.0,
            backend="heuristic",
        ),
        decision=ErasabilityDecision(True, "in_bubble", 0.9, False),
        erase_method="flat",
    )


def test_appearance_for_prepared_is_pure_and_has_rgb_text_color():
    prepared = _prepared_block()
    before_mask = prepared.mask_result.mask.copy()

    appearance = appearance_for_prepared(prepared)

    assert np.array_equal(prepared.mask_result.mask, before_mask)
    assert appearance["text_color"] == (0, 0, 0)
    assert appearance["erase_method"] == "stroke-fill-sampled"
    assert appearance["erase_mask_coverage"] == 0.25
    assert appearance["should_skip"] is False


def test_appearance_for_uncertain_prepared_block_skips_rendering():
    prepared = _prepared_block()
    prepared.decision = ErasabilityDecision(False, "risky_background", 0.2, True)
    prepared.erase_method = "preserve"

    appearance = appearance_for_prepared(prepared)

    assert appearance["should_skip"] is True
