import cv2
import numpy as np

from add_text import _analyze_region
from vision.region_analysis import analyze_region


def test_extracted_region_analysis_matches_legacy_wrapper_without_mutation():
    image = np.full((140, 180, 3), 255, np.uint8)
    cv2.ellipse(image, (90, 70), (65, 42), 0, 0, 360, (0, 0, 0), 3)
    original = image.copy()

    typed = analyze_region(image, (48, 48, 125, 82))
    legacy = _analyze_region(image, [48, 48, 125, 82])

    assert typed is not None
    assert legacy is not None
    assert legacy["bubble_context"] == typed.bubble_context
    assert legacy["uniformity"] == typed.uniformity
    assert legacy["mean_intensity"] == typed.mean_intensity
    assert legacy["is_bubble"] == (typed.bubble_context == "in_bubble")
    assert np.array_equal(image, original)


def test_region_analysis_rejects_an_empty_bbox():
    image = np.full((20, 20, 3), 255, np.uint8)

    assert analyze_region(image, (8, 8, 8, 12)) is None
