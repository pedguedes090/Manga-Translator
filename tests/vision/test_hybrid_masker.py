import cv2
import numpy as np

from vision.maskers.hybrid import HybridTextMasker
from vision.postprocess import (
    apply_bubble_gate,
    hysteresis_mask,
    keep_ocr_anchored_components,
)
from vision.region_analysis import analyze_region


def test_hysteresis_keeps_weak_pixels_connected_to_strong_seed_only():
    probability = np.array(
        [[0.7, 0.4, 0.0, 0.4], [0.0, 0.4, 0.0, 0.4]],
        np.float32,
    )

    mask = hysteresis_mask(probability, low=0.34, high=0.62)

    assert mask[:, :2].sum() == 3 * 255
    assert mask[:, 3].sum() == 0


def test_bubble_gate_reserves_a_safe_border_inside_the_bubble():
    mask = np.full((15, 15), 255, np.uint8)
    bubble = np.zeros_like(mask)
    cv2.rectangle(bubble, (2, 2), (12, 12), 255, -1)

    gated = apply_bubble_gate(mask, bubble, border_px=2)

    assert gated[3, 7] == 0
    assert gated[7, 7] == 255
    assert gated[0, 0] == 0


def test_component_filter_drops_shapes_outside_the_ocr_anchor():
    mask = np.zeros((20, 30), np.uint8)
    cv2.rectangle(mask, (2, 2), (7, 7), 255, -1)
    cv2.rectangle(mask, (14, 8), (20, 14), 255, -1)

    filtered = keep_ocr_anchored_components(mask, (12, 6, 24, 17))

    assert filtered[4, 4] == 0
    assert filtered[10, 16] == 255


def test_hybrid_masker_preserves_bubble_outline():
    image = np.full((140, 180, 3), 255, np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(
        image,
        "HEY",
        (52, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
    )
    bbox = (40, 35, 140, 95)
    region = analyze_region(image, bbox)
    assert region is not None

    result = HybridTextMasker().generate(image, bbox, "HEY", region, None)

    assert result.mask[35, 0] == 0
    assert result.coverage < 0.65
    assert result.backend == "hybrid"
    assert 1 <= result.debug["dilation_radius"] <= 4
