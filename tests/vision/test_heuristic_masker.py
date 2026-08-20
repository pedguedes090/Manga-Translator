import cv2
import numpy as np

from add_text import _build_text_stroke_mask
from vision.maskers.heuristic import HeuristicTextMasker
from vision.region_analysis import analyze_region


def test_heuristic_backend_matches_legacy_mask_pixels():
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
    bbox = (10, 20, 120, 65)
    region = analyze_region(image, bbox)
    assert region is not None

    result = HeuristicTextMasker().generate(image, bbox, "TEST", region, None)
    roi = image[20:65, 10:120]
    legacy = _build_text_stroke_mask(
        roi,
        region.mean_bgr,
        {"intensity_std": region.intensity_std, "uniformity": region.uniformity},
    )

    assert np.array_equal(result.mask, legacy)
    assert result.roi_bbox == bbox
    assert result.probability is None
    assert result.backend == "heuristic"
    assert result.coverage == np.count_nonzero(legacy) / legacy.size


def test_heuristic_backend_returns_an_empty_mask_for_blank_region():
    image = np.full((40, 50, 3), 240, np.uint8)
    bbox = (5, 6, 35, 30)
    region = analyze_region(image, bbox)
    assert region is not None

    result = HeuristicTextMasker().generate(image, bbox, "", region, None)

    assert not np.any(result.mask)
    assert result.confidence == 0.0
