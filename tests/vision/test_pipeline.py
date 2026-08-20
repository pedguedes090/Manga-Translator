from unittest.mock import Mock

import cv2
import numpy as np

from add_text import assess_erasability, erase_text_region
from vision.config import VisionConfig
from vision.maskers.hybrid import HybridTextMasker
from vision.pipeline import VisionPipeline, _choose_erase_method
from vision.types import (
    ErasabilityDecision,
    MaskResult,
    PreparedBlock,
    RegionAnalysis,
)


def _route_region(uniformity, bubble_context, texture_std, coverage=0.5):
    region = RegionAnalysis(
        mean_bgr=(220, 220, 220),
        mean_intensity=220.0,
        intensity_std=20.0,
        edge_score=30.0,
        texture_std=texture_std,
        dominant_tone="light",
        uniformity=uniformity,
        bubble_context=bubble_context,
    )
    mask_result = MaskResult(
        roi_bbox=(0, 0, 20, 20),
        mask=np.full((20, 20), 255, np.uint8),
        probability=None,
        bubble_mask=None,
        coverage=coverage,
        confidence=0.9,
        edge_touch_ratio=0.0,
        backend="heuristic",
    )
    decision = ErasabilityDecision(True, "accepted", 0.9, False)
    config = VisionConfig.load("configs/vision.json")
    return _choose_erase_method(region, mask_result, decision, config)


def test_large_mask_inside_bubble_uses_opencv_not_lama():
    assert _route_region("textured", "in_bubble", texture_std=35.0) == "telea"


def test_nonflat_uniform_background_uses_opencv_not_lama():
    assert (
        _route_region("uniform", "on_artwork_light", texture_std=20.0)
        == "telea"
    )


def test_textured_artwork_uses_opencv_even_for_large_mask():
    assert (
        _route_region("textured", "on_artwork_mixed", texture_std=35.0)
        == "telea"
    )


def _lama_block(block_id, roi_bbox):
    x1, y1, x2, y2 = roi_bbox
    region = RegionAnalysis(
        mean_bgr=(100, 100, 100),
        mean_intensity=100.0,
        intensity_std=60.0,
        edge_score=60.0,
        texture_std=60.0,
        dominant_tone="mid",
        uniformity="complex",
        bubble_context="on_artwork_mixed",
    )
    mask = np.full((y2 - y1, x2 - x1), 255, np.uint8)
    return PreparedBlock(
        block_id=block_id,
        text="",
        bbox=roi_bbox,
        region=region,
        mask_result=MaskResult(
            roi_bbox=roi_bbox,
            mask=mask,
            probability=None,
            bubble_mask=None,
            coverage=1.0,
            confidence=0.9,
            edge_touch_ratio=0.0,
            backend="heuristic",
        ),
        decision=ErasabilityDecision(True, "accepted_complex", 0.9, False),
        erase_method="lama_full_page",
    )


def test_erase_page_unions_complex_masks_for_one_inference():
    class RecordingInpainter:
        def __init__(self):
            self.masks = []

        def inpaint(self, image, mask):
            self.masks.append(mask.copy())
            return np.full_like(image, 255)

    inpainter = RecordingInpainter()
    pipeline = VisionPipeline(
        masker=HybridTextMasker(), lama_inpainter=inpainter
    )
    image = np.zeros((20, 30, 3), np.uint8)
    first = _lama_block("first", (2, 3, 8, 9))
    second = _lama_block("second", (20, 10, 27, 17))

    output, results = pipeline.erase_page(image, [first, second])

    assert len(inpainter.masks) == 1
    assert np.all(inpainter.masks[0][3:9, 2:8] == 255)
    assert np.all(inpainter.masks[0][10:17, 20:27] == 255)
    assert np.all(output[inpainter.masks[0] > 0] == 255)
    assert np.array_equal(output[inpainter.masks[0] == 0], image[inpainter.masks[0] == 0])
    assert [result.method for result in results] == [
        "lama_full_page",
        "lama_full_page",
    ]


def test_erase_page_reports_runtime_fallback_warning():
    class FallbackInpainter:
        def __init__(self):
            self.last_run = None

        def inpaint(self, image, mask):
            self.last_run = type(
                "Run", (), {"warning": "LaMa CUDA OOM; used Telea"}
            )()
            return image.copy()

    pipeline = VisionPipeline(
        masker=HybridTextMasker(), lama_inpainter=FallbackInpainter()
    )

    _, results = pipeline.erase_page(
        np.zeros((20, 30, 3), np.uint8),
        [_lama_block("complex", (2, 3, 8, 9))],
    )

    assert results[0].warning == "LaMa CUDA OOM; used Telea"


def test_erase_page_continues_with_telea_when_lama_runtime_fails():
    class BrokenInpainter:
        def inpaint(self, image, mask):
            raise RuntimeError("checkpoint is unavailable")

    image = np.zeros((20, 30, 3), np.uint8)
    cv2.rectangle(image, (0, 0), (29, 19), (80, 100, 120), -1)
    pipeline = VisionPipeline(
        masker=HybridTextMasker(), lama_inpainter=BrokenInpainter()
    )

    output, results = pipeline.erase_page(
        image, [_lama_block("complex", (2, 3, 8, 9))]
    )

    assert output.shape == image.shape
    assert "checkpoint is unavailable" in results[0].warning
    assert "Telea" in results[0].warning


def test_prepare_then_assess_and_erase_generates_one_mask():
    image = np.full((90, 150, 3), 245, np.uint8)
    cv2.putText(
        image,
        "TEST",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
    )
    masker = Mock(wraps=HybridTextMasker())
    pipeline = VisionPipeline(masker=masker, bubble_segmenter=None)

    prepared = pipeline.prepare_page(
        image, [{"text": "TEST", "bbox": [10, 20, 120, 65]}]
    )
    pipeline.assess(prepared[0])
    pipeline.erase_block(image.copy(), prepared[0])

    assert masker.generate.call_count == 1


def test_prepare_page_uses_stable_normalized_block_ids():
    image = np.full((40, 50, 3), 245, np.uint8)
    pipeline = VisionPipeline(masker=HybridTextMasker(), bubble_segmenter=None)

    prepared = pipeline.prepare_page(
        image, [{"text": "", "bbox": [35.2, 30.8, 5.1, 4.2]}]
    )

    assert prepared[0].bbox == (5, 4, 35, 31)
    assert prepared[0].block_id == "block-0000-5-4-35-31"


def test_legacy_wrappers_consume_a_prepared_block_without_regeneration():
    image = np.full((90, 150, 3), 245, np.uint8)
    cv2.putText(
        image,
        "TEST",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
    )
    masker = Mock(wraps=HybridTextMasker())
    pipeline = VisionPipeline(masker=masker)
    prepared = pipeline.prepare_page(
        image, [{"text": "TEST", "bbox": [10, 20, 120, 65]}]
    )[0]

    assessment = assess_erasability(
        image, prepared.bbox, text=prepared.text, prepared=prepared
    )
    erased, _, appearance = erase_text_region(
        image.copy(), prepared.bbox, prepared=prepared
    )

    assert assessment["safe"] == prepared.decision.safe
    assert appearance["erase_mask_coverage"] == prepared.mask_result.coverage
    assert erased.shape == image.shape
    assert masker.generate.call_count == 1


def test_prepare_page_runs_bubble_segmentation_once_for_all_blocks():
    image = np.full((70, 140, 3), 245, np.uint8)
    segmenter = Mock()
    segmenter.segment.return_value = []
    pipeline = VisionPipeline(
        masker=HybridTextMasker(), bubble_segmenter=segmenter
    )

    prepared = pipeline.prepare_page(
        image,
        [
            {"text": "A", "bbox": [10, 10, 50, 35]},
            {"text": "B", "bbox": [75, 30, 125, 60]},
        ],
    )

    assert len(prepared) == 2
    segmenter.segment.assert_called_once_with(image)
