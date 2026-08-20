from unittest.mock import Mock

import cv2
import numpy as np

from add_text import assess_erasability, erase_text_region
from vision.maskers.hybrid import HybridTextMasker
from vision.pipeline import VisionPipeline


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
