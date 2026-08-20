from pathlib import Path

import numpy as np

from vision.types import ErasabilityDecision, MaskResult, PreparedBlock, RegionAnalysis


def test_prepared_block_keeps_runtime_array_outside_serialized_summary():
    region = RegionAnalysis(
        mean_bgr=(245, 245, 245),
        mean_intensity=245.0,
        intensity_std=2.0,
        edge_score=4.0,
        texture_std=3.0,
        dominant_tone="light",
        uniformity="uniform",
        bubble_context="in_bubble",
    )
    mask = MaskResult(
        roi_bbox=(10, 20, 50, 60),
        mask=np.zeros((40, 40), np.uint8),
        probability=None,
        bubble_mask=None,
        coverage=0.0,
        confidence=0.9,
        edge_touch_ratio=0.0,
        backend="heuristic",
    )
    block = PreparedBlock(
        block_id="page-0-block-0",
        text="hello",
        bbox=(10, 20, 50, 60),
        region=region,
        mask_result=mask,
        decision=ErasabilityDecision(True, "uniform_background", 0.9, False),
        erase_method="flat",
        mask_ref=Path("vision/page-0-block-0.npz"),
    )

    summary = block.to_summary()

    assert summary["block_id"] == "page-0-block-0"
    assert summary["bbox"] == [10, 20, 50, 60]
    assert summary["mask_ref"] == "vision/page-0-block-0.npz"
    assert "mask" not in summary
